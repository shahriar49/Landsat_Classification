This project is aimed to do a state-of-the-art land cover classification using Landsat timer-series data. We employ all spectral, spatial, and temporal dimensions of data via a complex neural network consisting of below sections:

  1) A multilayer LSTM network which processes the recurrent input (sequence of features per sample point) and
      generates one set of features in the output
  2) A multilayer dense network which processes the annual and static input (one set of features per sample point)
  3) A multilayer dense network which processes the concatenation of the above two vectors and produce the final
      softmax layer and generated the predicted class
      
 Various files in this project are used to generate the input data, process it to the suitable TFRecord format, running the simulation, and processing the results.
