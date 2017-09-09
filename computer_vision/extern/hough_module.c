#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "Python.h"
#include "numpy/arrayobject.h"

#define PI 3.1415


/** Takes a PyArrayObject an returns a C array */
double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin);
/** Frees a C array */
void free_Carrayptrs(double **v);
/** Standard memory allocator for arrays */
double **ptrvector(long n);
/** Hough accumulator function */
static PyObject* hough_line (PyObject* self, PyObject* args);

/* Math Functions */
double* linspace (double a, double b, int n, double u[]);


static PyObject* hough_line (PyObject* self, PyObject* args)
{
    int nd;
    double **c_arr_in, **c_arr_out;

    PyObject *arg1  = NULL, *out = NULL;
    PyArrayObject *arr = NULL, *oarr = NULL;

    if (!PyArg_ParseTuple (args, "OO!", &arg1, &PyArray_Type, &out))
        return NULL;

    arr = (PyArrayObject*)PyArray_FROM_OTF (arg1, NPY_DOUBLE, NPY_IN_ARRAY);
    if (arr == NULL) return NULL;

    npy_intp* dims_arr = PyArray_DIMS(arr);
    c_arr_in  = pymatrix_to_Carrayptrs(arr);

    int max_distance = sqrt((int)pow(dims_arr[0], 2) + (int)pow(dims_arr[1], 2));
    int H[180][max_distance];
    for (int i = 0; i < 180; i++)
        for (int j = 0; j < max_distance; j++)
            H[i][j] = 0;

    oarr = (PyArrayObject*)PyArray_FROM_OTF (out, NPY_DOUBLE, NPY_IN_ARRAY);
    if (arr == NULL) goto fail;

    nd = PyArray_NDIM(arr);
    npy_intp* dims_out = PyArray_DIMS(out);
    c_arr_out = pymatrix_to_Carrayptrs(oarr);

    if (dims_arr[0] != dims_out[0] || dims_arr[1] != dims_out[1]) {
        PyErr_SetString(PyExc_ValueError, "arrays must be same size");
        goto fail;
    }


    // Calculation is here
    double thetas[180];
    linspace(- PI / 2, PI / 2, 180, thetas);

    for (int x = 0; x < dims_arr[0]; x++) {
        for (int y = 0; y < dims_out[1]; y++) {
            for (int t = 0; t < 180; t++) {
                int theta = (int)floor(atan2(y,x));
                int rho   = (int)floor(x * cos(theta) + y * sin(theta));
                H[theta][rho] += 1;
            }
        }
    }

    free_Carrayptrs(c_arr_in);
    free_Carrayptrs(c_arr_out);
    Py_DECREF(arr);
    Py_DECREF(oarr);
    return Py_None;

fail:
    Py_XDECREF(arr);
    PyArray_XDECREF_ERR(oarr);
    return NULL;
}

static PyObject* version (PyObject* self)
{
    return Py_BuildValue ("s", "Version 0.0");
}

/**
 * Method definitions of the module
 **/
static PyMethodDef hough_methods[] = {
    {"hough_line", hough_line, METH_VARARGS, "haugh accumulator"},
    {"version", (PyCFunction)version, METH_NOARGS, "Version of the module"},
    {NULL, NULL, 0, NULL} // must have this last arg!
};

/**
 * Definition of the module
 **/
static struct PyModuleDef hough_module = {
    PyModuleDef_HEAD_INIT,
    "hough_module",
    "Hough Transforms to a numpy array",
    -1,
    hough_methods
};

// Init function name mush have the format:
//      PyInit_ + module name
PyMODINIT_FUNC PyInit_hough_module (void)
{
    import_array();
    return PyModule_Create (&hough_module);
}

// Math functions =========================
double* linspace (double a, double b, int n, double u[])
{
    double c;
    int i;

    if (n < 2 || u == 0)
        return (void*)0;

    /* step size */
    c = (b - a)/(n - 1);

    /* fill vector */
    for (i = 0; i < n - 1; ++i)
        u[i] = a + i*c;

    /* fix last entry to b */
    u[n - 1] = b;

    return u;
}

// Util functions =========================
double **ptrvector(long n)
{
   double **v;
   v = (double **)malloc((size_t) (n*sizeof(double)));
   if (!v)   {
      printf("In **ptrvector. Allocation of memory for double array failed.");
      exit(0);
   }
   return v;
}


double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin)
{
   double **c, *a;
   int i,n,m;

   n=arrayin->dimensions[0];
   m=arrayin->dimensions[1];
   c=ptrvector(n);
   a=(double *) arrayin->data; /* pointer to arrayin data as double */
   for ( i=0; i<n; i++) {
      c[i]=a+i*m;
   }
   return c;
}


void free_Carrayptrs(double **v)
{
   free((char*) v);
}
