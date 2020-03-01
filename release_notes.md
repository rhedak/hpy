# Release Notes

## v0.1.3
- hhpy.regression (new module)
  - Conv1DNN model: wrapper for keras.Conv1D
- hhpy.main
  - new functions
    - copy_function: returns a copy of a function (useful for defining repr rules) 
  - changes:
    - renamed 'force_list' to 'assert_list' for consistency
    - rename 'force_scalar' to 'assert_scalar' for consistency
  - bug fixes:
    - fixed repr
- hhpy.ds
  - new functions
    - rolling_lfit: calculate the rolling linear fit over a pandas Series
    - rank: rank a DataFrame column / Series according to criteria, supports groupby
- hhpy.plotting
  - new functions
    - cat_to_color: Encodes a categorical column as colors of a specified palette
- hhpy.modelling
  - k_split, train, predict and fit now support fit_type 'k_cross'
    - automatically split the data into k sub splits and cross validate
    - predict can choose between k_predict_type 'test' (return test prediction for each k group) and 'all' returns
      prediction for each k group
  - new functions
    - to_keras_3d: Turns a pandas DataFrame into a 3D array required by Keras

## v0.1.2
- new function: rolling_lfit -> calculate the rolling linear fit over a pandas Series
- reworked BaseClass and added generic (internal) _get_repr function for better extension methods
- get_repr supports rules for extending standard repr
- a lot of bugfixes
- documentation updates

## v0.1.1
- new class: DFMapping, creates a mapping from that unifies conventions for a DataFrame
- first unit tests
- documentation updates
- bugfixes

## v0.1.0
The documentation has been updated and many examples have been added. Bugs have been fixed in a number of functions. 
From the next version on I'm planning on having proper release notes...