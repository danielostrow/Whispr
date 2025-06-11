#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject *separate_by_segmentation_c(PyObject *self, PyObject *args) {
    PyArrayObject *signal_obj;
    PyObject *segments_obj, *labels_obj;
    int sr;
    
    // Parse Python arguments
    if (!PyArg_ParseTuple(args, "O!iOO", &PyArray_Type, &signal_obj, &sr, 
                          &segments_obj, &labels_obj)) {
        return NULL;
    }
    
    // Ensure input is a contiguous array of the right type
    PyArrayObject *signal = (PyArrayObject *) PyArray_GETCONTIGUOUS(signal_obj);
    if (signal == NULL) return NULL;
    
    // Get signal dimensions
    npy_intp signal_len = PyArray_DIM(signal, 0);
    float *signal_data = (float *) PyArray_DATA(signal);
    
    // Verify segments and labels are lists and have the same length
    if (!PyList_Check(segments_obj) || !PyList_Check(labels_obj)) {
        Py_DECREF(signal);
        PyErr_SetString(PyExc_TypeError, "segments and labels must be lists");
        return NULL;
    }
    
    Py_ssize_t num_segments = PyList_Size(segments_obj);
    if (num_segments != PyList_Size(labels_obj)) {
        Py_DECREF(signal);
        PyErr_SetString(PyExc_ValueError, "segments and labels must have the same length");
        return NULL;
    }
    
    // Calculate frame and hop lengths from sr
    int frame_len = (int)(sr * 0.025); // 25 ms
    int hop_len = (int)(sr * 0.01);    // 10 ms
    
    // Create a dictionary to collect segments per speaker
    PyObject *speakers_dict = PyDict_New();
    if (!speakers_dict) {
        Py_DECREF(signal);
        return NULL;
    }
    
    // Group segments by speaker label
    for (Py_ssize_t i = 0; i < num_segments; i++) {
        PyObject *segment = PyList_GetItem(segments_obj, i);
        PyObject *label = PyList_GetItem(labels_obj, i);
        
        // Convert label to string key
        int label_int = PyLong_AsLong(label);
        PyObject *key = PyUnicode_FromFormat("Speaker_%d", label_int + 1);
        
        // Get or create the list for this speaker
        PyObject *speaker_segments = PyDict_GetItem(speakers_dict, key);
        if (!speaker_segments) {
            speaker_segments = PyList_New(0);
            PyDict_SetItem(speakers_dict, key, speaker_segments);
            Py_DECREF(speaker_segments);  // Dict owns the reference now
        }
        
        // Append segment
        PyList_Append(speaker_segments, segment);
        Py_DECREF(key);
    }
    
    // Create list of speaker dicts to return
    PyObject *result_list = PyList_New(0);
    if (!result_list) {
        Py_DECREF(signal);
        Py_DECREF(speakers_dict);
        return NULL;
    }
    
    // Process each speaker
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    
    while (PyDict_Next(speakers_dict, &pos, &key, &value)) {
        PyObject *time_segments = PyList_New(0);
        PyObject *audio_slices = PyList_New(0);
        
        // Process each segment for this speaker
        Py_ssize_t num_speaker_segments = PyList_Size(value);
        for (Py_ssize_t i = 0; i < num_speaker_segments; i++) {
            PyObject *segment = PyList_GetItem(value, i);
            
            // Extract start and end frame
            int start_frame = PyLong_AsLong(PyTuple_GetItem(segment, 0));
            int end_frame = PyLong_AsLong(PyTuple_GetItem(segment, 1));
            
            // Convert to sample indices
            int start_sample = start_frame * hop_len;
            int end_sample = end_frame * hop_len + frame_len;
            
            // Ensure bounds are valid
            if (end_sample > signal_len) end_sample = signal_len;
            
            // Extract audio slice
            npy_intp slice_dims[1] = {end_sample - start_sample};
            PyArrayObject *slice = (PyArrayObject *) PyArray_SimpleNew(1, slice_dims, NPY_FLOAT32);
            if (!slice) continue;
            
            // Copy the audio data
            float *slice_data = (float *) PyArray_DATA(slice);
            memcpy(slice_data, signal_data + start_sample, slice_dims[0] * sizeof(float));
            
            // Add to audio slices
            PyList_Append(audio_slices, (PyObject *)slice);
            Py_DECREF(slice);
            
            // Convert to time and add to time segments
            double start_time = (double)start_sample / sr;
            double end_time = (double)end_sample / sr;
            PyObject *time_segment = Py_BuildValue("(dd)", start_time, end_time);
            PyList_Append(time_segments, time_segment);
            Py_DECREF(time_segment);
        }
        
        // Concatenate audio slices if any
        if (PyList_Size(audio_slices) > 0) {
            // First calculate total length
            npy_intp total_length = 0;
            for (Py_ssize_t i = 0; i < PyList_Size(audio_slices); i++) {
                PyArrayObject *slice = (PyArrayObject *)PyList_GetItem(audio_slices, i);
                total_length += PyArray_DIM(slice, 0);
            }
            
            // Create concatenated array
            npy_intp concat_dims[1] = {total_length};
            PyArrayObject *concatenated = (PyArrayObject *) PyArray_SimpleNew(1, concat_dims, NPY_FLOAT32);
            if (!concatenated) continue;
            
            // Copy slices to concatenated array
            float *concat_data = (float *) PyArray_DATA(concatenated);
            npy_intp offset = 0;
            
            for (Py_ssize_t i = 0; i < PyList_Size(audio_slices); i++) {
                PyArrayObject *slice = (PyArrayObject *)PyList_GetItem(audio_slices, i);
                npy_intp slice_len = PyArray_DIM(slice, 0);
                float *slice_data = (float *) PyArray_DATA(slice);
                
                memcpy(concat_data + offset, slice_data, slice_len * sizeof(float));
                offset += slice_len;
            }
            
            // Create speaker dict
            PyObject *speaker_dict = PyDict_New();
            PyDict_SetItemString(speaker_dict, "id", key);
            PyDict_SetItemString(speaker_dict, "signal", (PyObject *)concatenated);
            PyDict_SetItemString(speaker_dict, "sr", PyLong_FromLong(sr));
            PyDict_SetItemString(speaker_dict, "segments", time_segments);
            
            // Add to result list
            PyList_Append(result_list, speaker_dict);
            
            // Clean up
            Py_DECREF(concatenated);
            Py_DECREF(speaker_dict);
        }
        
        Py_DECREF(time_segments);
        Py_DECREF(audio_slices);
    }
    
    // Clean up
    Py_DECREF(signal);
    Py_DECREF(speakers_dict);
    
    return result_list;
}

// Module methods
static PyMethodDef SeparationMethods[] = {
    {"separate_by_segmentation_c", separate_by_segmentation_c, METH_VARARGS, 
     "Separate audio by segmentation (C implementation)"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef separation_module = {
    PyModuleDef_HEAD_INIT,
    "separation_c",
    "C implementation of audio separation functions",
    -1,
    SeparationMethods
};

// Initialize module
PyMODINIT_FUNC PyInit_separation_c(void) {
    PyObject *m;
    
    m = PyModule_Create(&separation_module);
    if (m == NULL)
        return NULL;
    
    // Import NumPy C-API
    import_array();
    
    return m;
} 