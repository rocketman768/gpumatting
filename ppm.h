#include <string.h>
#include <stdio.h>

/*!
 * \brief Reads a binary or ascii PGM (grayscale image) file.
 * 
 * \param filename The pgm file to read
 * \param w Output width
 * \param h Output height
 * \returns a pointer to the image in row-major format. The calling
 *          function has responsibility to free the memory. A NULL is returned
 *          in the case of failure.
 */
unsigned char* pgmread(char* filename, int* w, int* h)
{
    FILE* file;
    char line[256];
    int maxval;
    int binary;
    int nread;
    int numpix;
    int i,j,k,int_tmp;

    unsigned char* data;
    
    if ((file = fopen(filename, "r")) == NULL)
    {
       printf("ERROR: file open failed\n");
       *h = *w = 0;
       return(NULL);
    }
    fgets(line, 256, file);
    if (strncmp(line,"P5", 2))
    {
       if (strncmp(line,"P2", 2))
       {
          printf("pgm read: not a pgm file\n");
          *h = *w = 0;
          return(NULL);
       }
       else 
          binary = 0;
    }
    else 
       binary = 1;

    fgets(line, 256, file);
    while (line[0] == '#')
       fgets(line, 256, file);

    sscanf(line,"%d %d", w, h);
    fgets(line, 256, file);
    sscanf(line, "%d", &maxval);
    
    numpix = (*w)*(*h);
    
    if ((data = (unsigned char*)calloc(numpix, sizeof(unsigned char*))) == NULL)
    {
      printf("Memory allocation error. Exit program");
      exit(1);
    }

    if (binary)
    {
       nread = fread( (void*)data, sizeof(unsigned char), numpix, file);
       if( nread != numpix )
       {
         fprintf(stderr, "Error: read %d/%d pixels.", nread, numpix);
         exit(1);
       }
    }
    else
    {
       k = 0;
       for (i = 0; i < (*h); i++)
       {
          for (j = 0; j < (*w); j++)
          {
             fscanf(file, "%d", &int_tmp);
             data[k++] = (unsigned char)int_tmp;
          }  
       }
    }
    
    fclose(file);
    return data;
}

/*!
 * \brief Reads a binary or ascii PPM (rgb image) file.
 * 
 * \param filename The ppm file to read
 * \param w Output width
 * \param h Output height
 * \param maxval Output maximum value.
 * \returns a pointer to the image in row-major RGB format. The calling
 *          function has responsibility to free the memory. A NULL is returned
 *          in the case of failure.
 */
unsigned char* ppmread(char* filename, int* w, int* h, int* maxval)
{
    FILE* file;
    char line[256];
    int binary;
    int nread;
    int numpix;
    int i,j,k,int_tmp;

    unsigned char* data;
    
    if ((file = fopen(filename, "r")) == NULL)
    {
       printf("ERROR: file open failed\n");
       *h = *w = 0;
       return(NULL);
    }
    fgets(line, 256, file);
    if (strncmp(line,"P6", 2))
    {
       if (strncmp(line,"P3", 2))
       {
          printf("ppm read: not a ppm file\n");
          *h = *w = 0;
          return(NULL);
       }
       else 
          binary = 0;
    }
    else 
       binary = 1;

    fgets(line, 256, file);
    while (line[0] == '#')
       fgets(line, 256, file);

    sscanf(line,"%d %d", w, h);
    fgets(line, 256, file);
    sscanf(line, "%d", maxval);
    
    if( *maxval < 0 || *maxval > 255 )
    {
       fprintf(stderr, "Error: maximum value %d is bad.\n", *maxval);
       exit(1);
    }
    
    numpix = (*w)*(*h);
    
    if ((data = (unsigned char*)calloc(numpix*3, sizeof(unsigned char*))) == NULL)
    {
      printf("Memory allocation error. Exit program");
      exit(1);
    }

    if (binary)
    {
       nread = fread( (void*)data, sizeof(unsigned char), numpix*3, file);
       if( nread != numpix*3 )
       {
          fprintf(stderr, "Error: read %d/%d pixels.", nread/3, numpix);
          exit(1);
       }
    }
    else
    {
       k = 0;
       for (i = 0; i < (*h); i++)
       {
          for (j = 0; j < (*w)*3; j++)
          {
             fscanf(file, "%d", &int_tmp);
             data[k++] = (unsigned char)int_tmp;
          }  
       }
    }
    
    fclose(file);
    return data;
}

/*!
 * \brief Read a normalized floating-point image.
 * 
 * \sa ppmread()
 */
float* ppmread_float(char* filename, int* w, int* h )
{
   int maxval;
   int i, numpix;
   unsigned char* cdata;
   float* fdata;
   
   cdata = ppmread(filename, w, h, &maxval);
   numpix = (*w)*(*h);
   fdata = (float*)malloc( 3*numpix*sizeof(float) );
   
   for( i = 0; i < numpix*3; ++i )
      fdata[i] = static_cast<float>(cdata[i])/maxval;
   
   free(cdata);
   return fdata;
}

/*!
 * \brief Write a PGM image.
 * 
 * \param filename The file to write to.
 * \param w Image width
 * \param h Image height
 * \param data Row-major image data
 * \param comment_string Comments (NULL if none)
 * \param binsave 1 for binary writing, 0 for text writing
 */
int pgmwrite(
   char* filename,
   int w, int h,
   unsigned char* data, 
   const char* comment_string,
   int binsave
)
{
    FILE* file;
    int maxval;
    int nread;
    int i,j,k;
    int numpix = w*h;
    
    if ((file = fopen(filename, "w")) == NULL)
    {
       printf("ERROR: file open failed\n");
       return(-1);
    }

    if (binsave == 1)
      fprintf(file,"P5\n");
    else
      fprintf(file,"P2\n");

    if (comment_string != NULL)
      fprintf(file,"# %s \n", comment_string);

    fprintf(file,"%d %d \n", w, h);
    
    maxval = 0;
    k = 0;
    for (i = 0; i < h; i++)
    {
       for (j=0; j < w; j++)
       {
         if ((int)data[k] > maxval)
            maxval = (int)data[k];
         ++k;
       }
    }

    fprintf(file, "%d \n", maxval);
    
    if (binsave == 1)
    {
      nread = fwrite(data, sizeof(unsigned char), numpix, file);
      if( nread != numpix )
      {
         fprintf(stderr, "Error: wrote %d/%d pixels.", nread, numpix);
         exit(1);
      }
    }
    else
    {
      printf("Writing to %s as ascii.\n", filename);

      k = 0;
      for(i=0; i<h; i++)
        for(j=0; j<w; j++)
          fprintf(file,"%d ", (int)data[k++]);
    }     
   
    fclose(file);
    return 0;
}
