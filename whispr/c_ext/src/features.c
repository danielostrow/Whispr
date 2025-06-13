#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

/* Define M_PI if it's not defined (Windows) */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Check for OpenMP */
#ifdef _OPENMP
#include <omp.h>
#endif

// Create Hanning window
static void create_hanning_window(float *window, int length) {
    for (int i = 0; i < length; i++) {
        window[i] = 0.5 * (1 - cos(2 * M_PI * i / (length - 1)));
    }
}

static PyObject *compute_frame_energy(PyObject *self, PyObject *args) {
    PyArrayObject *frames_obj;
    
    // Parse Python arguments
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &frames_obj)) {
        return NULL;
    }
    
    // Ensure input is a contiguous array of the right type
    PyArrayObject *frames = (PyArrayObject *) PyArray_GETCONTIGUOUS(frames_obj);
    if (frames == NULL) return NULL;
    
    // Get dimensions
    npy_intp num_frames = PyArray_DIM(frames, 0);
    npy_intp frame_length = PyArray_DIM(frames, 1);
    
    // Create Hanning window
    float *window = (float *) malloc(frame_length * sizeof(float));
    if (!window) {
        Py_DECREF(frames);
        return PyErr_NoMemory();
    }
    create_hanning_window(window, frame_length);
    
    // Create output array for energies
    npy_intp dims[1] = {num_frames};
    PyArrayObject *energies = (PyArrayObject *) PyArray_ZEROS(1, dims, NPY_FLOAT32, 0);
    if (energies == NULL) {
        free(window);
        Py_DECREF(frames);
        return NULL;
    }
    
    // Compute energy for each frame
    float *frames_data = (float *) PyArray_DATA(frames);
    float *energies_data = (float *) PyArray_DATA(energies);
    
    // Convert to int for loop iterations
    int num_frames_int = (int)num_frames;
    
    /* Only use OpenMP directive if OpenMP is available */
    #ifdef _OPENMP
    #pragma omp parallel for
    for (int i = 0; i < num_frames_int; i++) {
    #else
    for (npy_intp i = 0; i < num_frames; i++) {
    #endif
        float energy = 0.0;
        float *frame = frames_data + i * frame_length;
        
        for (npy_intp j = 0; j < frame_length; j++) {
            float windowed_sample = frame[j] * window[j];
            energy += windowed_sample * windowed_sample;
        }
        
        energies_data[i] = energy;
    }
    
    // Clean up
    free(window);
    Py_DECREF(frames);
    
    return (PyObject *) energies;
}

// Module methods
static PyMethodDef FeatureMethods[] = {
    {"compute_frame_energy", compute_frame_energy, METH_VARARGS, 
     "Compute energy of each frame with Hanning window (C implementation)"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef features_module = {
    PyModuleDef_HEAD_INIT,
    "features_c",
    "C implementation of audio feature extraction",
    -1,
    FeatureMethods
};

// Initialize module
PyMODINIT_FUNC PyInit_features_c(void) {
    PyObject *m;
    
    m = PyModule_Create(&features_module);
    if (m == NULL)
        return NULL;
    
    // Import NumPy C-API
    import_array();
    
    return m;
} 