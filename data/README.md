## Note

The raw flight trajectories used in this work are not available since the trajectory sequences are collected from a 
real-world ATC system and contain a large amount of position and route information, sharing these sensitive data could 
pose a potential risk to the safety of civil aviation operations. Nonetheless, the processed example samples are 
available in `example_data.npy`. Source data for figures are also provided in `source_data.xlsx`.

## Guidance of example data

The example data has been stored as a binary file in NumPy format. It can be accessed by using:

```python
import numpy as np
example_data = np.load("example_data.npy") 
```

The first dimension of `example_data` is the sample number of 500. The second is the sliding-window size of 10 (the first 9
lines represent the input trajectory sequence, whereas the final line serves as the target trajectory point). And the 
last dimension of 6 indicates the six attributes: longitude (degree), latitude (degree), altitude (10 meters), and 
velocities (kilometers per hour) along previous three position components. 

The predicted trajectory of example data can be found in `desired_results.xlsx`. 
