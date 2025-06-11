#include <Python.h>
#include <numpy/arrayobject.h>
#include <float.h>

// Helper function to find median of array
static float find_median(float *arr, npy_intp length) {
    // Make a copy to avoid modifying original
    float *temp = (float *) malloc(length * sizeof(float));
    if (!temp) return 0.0;
    
    memcpy(temp, arr, length * sizeof(float));
    
    // Simple insertion sort (sufficient for typical frame counts)
    for (npy_intp i = 1; i < length; i++) {
        float key = temp[i];
        npy_intp j = i - 1;
        
        while (j >= 0 && temp[j] > key) {
            temp[j + 1] = temp[j];
            j--;
        }
        temp[j + 1] = key;
    }
    
    // Find median
    float median;
    if (length % 2 == 0) {
        median = (temp[length/2 - 1] + temp[length/2]) / 2.0;
    } else {
        median = temp[length/2];
    }
    
    free(temp);
    return median;
}

static PyObject *simple_energy_vad_c(PyObject *self, PyObject *args) {
    PyArrayObject *energies_obj;
    float vad_energy_threshold;
    int min_frames;
    
    // Parse Python arguments
    if (!PyArg_ParseTuple(args, "O!fi", &PyArray_Type, &energies_obj, 
                          &vad_energy_threshold, &min_frames)) {
        return NULL;
    }
    
    // Ensure input is a contiguous array of the right type
    PyArrayObject *energies = (PyArrayObject *) PyArray_GETCONTIGUOUS(energies_obj);
    if (energies == NULL) return NULL;
    
    // Get dimensions
    npy_intp length = PyArray_DIM(energies, 0);
    float *energy_data = (float *) PyArray_DATA(energies);
    
    // Calculate median and threshold
    float median = find_median(energy_data, length);
    float threshold = median * (1.0 + vad_energy_threshold);
    
    // Create a boolean array for voiced frames
    bool *voiced = (bool *) malloc(length * sizeof(bool));
    if (!voiced) {
        Py_DECREF(energies);
        return PyErr_NoMemory();
    }
    
    // Fill the voiced array
    for (npy_intp i = 0; i < length; i++) {
        voiced[i] = energy_data[i] > threshold;
    }
    
    // First pass to count segments
    int segment_count = 0;
    int start = -1;
    
    for (npy_intp i = 0; i < length; i++) {
        if (voiced[i] && start == -1) {
            start = i;
        } else if (!voiced[i] && start != -1) {
            start = -1;
            segment_count++;
        }
    }
    
    // Add final segment if needed
    if (start != -1) {
        segment_count++;
    }
    
    // Pre-allocate a Python list to hold tuples
    PyObject *segments = PyList_New(0);
    if (!segments) {
        free(voiced);
        Py_DECREF(energies);
        return NULL;
    }
    
    // Second pass to extract segments
    start = -1;
    for (npy_intp i = 0; i < length; i++) {
        if (voiced[i] && start == -1) {
            start = i;
        } else if (!voiced[i] && start != -1) {
            int end = i - 1;
            // Only add segments longer than min_frames
            if (end - start + 1 >= min_frames) {
                PyObject *tuple = Py_BuildValue("(ii)", start, end);
                PyList_Append(segments, tuple);
                Py_DECREF(tuple);
            }
            start = -1;
        }
    }
    
    // Handle last segment if needed
    if (start != -1) {
        int end = length - 1;
        if (end - start + 1 >= min_frames) {
            PyObject *tuple = Py_BuildValue("(ii)", start, end);
            PyList_Append(segments, tuple);
            Py_DECREF(tuple);
        }
    }
    
    // Clean up
    free(voiced);
    Py_DECREF(energies);
    
    return segments;
}

// Module methods
static PyMethodDef VadMethods[] = {
    {"simple_energy_vad_c", simple_energy_vad_c, METH_VARARGS, 
     "Voice Activity Detection based on energy (C implementation)"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef vad_module = {
    PyModuleDef_HEAD_INIT,
    "vad_c",
    "C implementation of Voice Activity Detection",
    -1,
    VadMethods
};

// Initialize module
PyMODINIT_FUNC PyInit_vad_c(void) {
    PyObject *m;
    
    m = PyModule_Create(&vad_module);
    if (m == NULL)
        return NULL;
    
    // Import NumPy C-API
    import_array();
    
    return m;
} 