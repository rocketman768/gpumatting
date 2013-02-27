
void downsampleOps(..., float const* im, int imW, int imH, int const* labels, int nlabels, int nlevels)
{
   // Coordinate matrix representation of a downsample operator.
   int* i;
   int* j;
   int* k;
   
   int numpix = imW*imH;
   int u,v;
   int curLabel;
   int prevNlabels;
   
   float tmp;
   float min;
   int minU, minV;
   int *__restrict__ mapping = (int*)malloc(nlabels*sizeof(int));
   int *__restrict__ merged = (int*)malloc(nlabels*sizeof(int));
   float *__restrict__ diffs = (float*)malloc(nlabels*nlabels*sizeof(float));
   float *__restrict__ means = (float*)malloc(3*nlabels*sizeof(float));
   int *__restrict__ segSize = (int*)malloc(nlabels*sizeof(int));
   memset(segSize, 0x00, nlabels*sizeof(int));
   ...
   
   // First level--------------------------------------------------------------
   memset(means, 0x00, 3*nlabels*sizeof(float));
   for( u = 0; u < numpix; ++u )
   {
      means[3*labels[u]+0] += im[4*u+0];
      means[3*labels[u]+1] += im[4*u+1];
      means[3*labels[u]+2] += im[4*u+2];
      
      ++segSize[labels[u]];
      i[u] = labels[u];
      j[u] = u;
      k[u] = 1.0f;
   }
   for( u = 0; u < nlabels; ++u )
      means[3*u+0] /= segSize[u]; means[3*u+1] /= segSize[u]; means[3*u+2] /= segSize[u];
   
   ...
   
   // Subsequent levels--------------------------------------------------------
   for( --nlevels; nlevels > 0; --nlevels )
   {
      prevNlabels = nlabels;
      curLabel = 0;
      memset(merged, 0x00, nlabels*sizeof(int));
      
      // diffs[u][v] = ||means[u]-means[v]||_2^2
      min = 1e6;
      for( u = 0; u < prevNlabels; ++u )
      {
         for( v = u+1, v < prevNlabels; ++v )
         {
            tmp = (means[3*u+0]-means[3*v+0]);
            tmp *= tmp;
            diffs[v + u * nlabels] = tmp;
            
            tmp = (means[3*u+1]-means[3*v+1]);
            tmp *= tmp;
            diffs[v + u * nlabels] += tmp;
            
            tmp = (means[3*u+2]-means[3*v+2]);
            tmp *= tmp;
            diffs[v + u * nlabels] += tmp;
            
            // Get location of minimum difference.
            if( diffs[v + u * nlabels] < min )
            {
               min = diffs[v + u * nlabels]
               minU = u;
               minV = v;
            }
         }
      }
      
      merged[minU] = 1;
      merged[minV] = 1;
      mapping[minU] = curLabel;
      mapping[minV] = curLabel;
      
      ++curLabel;
      
      while( true )
      {
         min = 1e6;
         for( u = 0; u < prevNlabels; ++u )
         {
            if( merged[u] )
               continue;
            
            for( v = u+1, v < prevNlabels; ++v )
            {
               if( merged[v] )
                  continue;
               
               // Get location of minimum difference.
               if( diffs[v + u * nlabels] < min )
               {
                  min = diffs[v + u * nlabels]
                  minU = u;
                  minV = v;
               }
            }
         }
         
         // This condition means exp( -||mean[u]-mean[v]||_2 ) < 0.90.
         if( min > 0.0111f )
            break;
         
         merged[minU] = 1;
         merged[minV] = 1;
         mapping[minU] = curLabel;
         mapping[minV] = curLabel;
         
         ++curLabel;
      }
      
      // Finish the mapping.
      for( u = 0; u < prevNlabels; ++u )
      {
         if( merged[u] )
            i[u] = mapping[u];
         else
            i[u] = curLabel++;
         j[u] = u;
         k[u] = 1.0f;
      }
      
      ...
      
      // NOTE: need to update means[] here.
      
      prevNlabels = curLabel;
   }
}