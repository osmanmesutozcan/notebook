#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "Python.h"
#include "numpy/arrayobject.h"


/** Takes a PyArrayObject an returns a C array */
double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin);
/** Frees a C array */
void free_Carrayptrs(double **v);
/** Standard memory allocator for arrays */
double **ptrvector(long n);
/** Hough accumulator function */
static PyObject* hough_line (PyObject* self, PyObject* args);


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
    oarr = (PyArrayObject*)PyArray_FROM_OTF (out, NPY_DOUBLE, NPY_IN_ARRAY);
    if (arr == NULL) goto fail;


// End Parse Args =========================
    nd = PyArray_NDIM(arr);
    npy_intp* dims_arr = PyArray_DIMS(arr);
    npy_intp* dims_out = PyArray_DIMS(out);

    c_arr_in  = pymatrix_to_Carrayptrs(arr);
    c_arr_out = pymatrix_to_Carrayptrs(oarr);

    if (dims_arr[0] != dims_out[0] || dims_arr[1] != dims_out[1]) {
        PyErr_SetString(PyExc_ValueError, "arrays must be same size");
        goto fail;
    }

    // Calculation is here
    for (int x = 0; x < dims_arr[0]; x++)
        for (int y = 0; y < dims_out[1]; y++) {
            double theta = atan2(y,x);
            double rho   = x * cos(theta) + y * sin(theta);

            // TODO: contruct return vals
            c_arr_out[x][y] = c_arr_in[x][y];

            printf("data  ==> %f\n", c_arr_in[x][y]);
            printf("theta ==> %f\n", theta);
            printf("rho   ==> %f\n\n", rho);
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


// Util functions =========================
double **ptrvector(long n) {
   double **v;
   v = (double **)malloc((size_t) (n*sizeof(double)));
   if (!v)   {
      printf("In **ptrvector. Allocation of memory for double array failed.");
      exit(0);
   }
   return v;
}


double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin) {
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


void free_Carrayptrs(double **v) {
   free((char*) v);
}
