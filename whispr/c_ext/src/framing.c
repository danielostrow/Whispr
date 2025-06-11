#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject *frame_signal_c(PyObject *self, PyObject *args) {
    PyArrayObject *signal_obj;
    int frame_length, hop_length;
    
    // Parse Python arguments
    if (!PyArg_ParseTuple(args, "O!ii", &PyArray_Type, &signal_obj, &frame_length, &hop_length)) {
        return NULL;
    }
    
    // Ensure input is a contiguous array of the right type
    PyArrayObject *signal = (PyArrayObject *) PyArray_GETCONTIGUOUS(signal_obj);
    if (signal == NULL) return NULL;
    
    // Get signal dimensions
    npy_intp signal_len = PyArray_DIM(signal, 0);
    
    // Calculate number of frames
    int num_frames = 1 + (signal_len - frame_length) / hop_length;
    if (num_frames <= 0) {
        Py_DECREF(signal);
        PyErr_SetString(PyExc_ValueError, "Signal is too short for the given frame_length");
        return NULL;
    }
    
    // Create output array dimensions
    npy_intp dims[2] = {num_frames, frame_length};
    PyArrayObject *frames = (PyArrayObject *) PyArray_ZEROS(2, dims, NPY_FLOAT32, 0);
    if (frames == NULL) {
        Py_DECREF(signal);
        return NULL;
    }
    
    // Copy data frame by frame (optimized memory access)
    float *signal_data = (float *) PyArray_DATA(signal);
    float *frames_data = (float *) PyArray_DATA(frames);
    
    for (int i = 0; i < num_frames; i++) {
        int start_idx = i * hop_length;
        // Use memcpy for efficient copying
        memcpy(frames_data + i * frame_length, signal_data + start_idx, 
               frame_length * sizeof(float));
    }
    
    // Clean up
    Py_DECREF(signal);
    
    return (PyObject *) frames;
}

// Module methods
static PyMethodDef FramingMethods[] = {
    {"frame_signal_c", frame_signal_c, METH_VARARGS, 
     "Frame a 1-D signal into overlapping frames (C implementation)"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef framing_module = {
    PyModuleDef_HEAD_INIT,
    "framing_c",
    "C implementation of audio framing functions",
    -1,
    FramingMethods
};

// Initialize module
PyMODINIT_FUNC PyInit_framing_c(void) {
    PyObject *m;
    
    m = PyModule_Create(&framing_module);
    if (m == NULL)
        return NULL;
    
    // Import NumPy C-API
    import_array();
    
    return m;
} 