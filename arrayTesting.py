import numpy as np
from speechLibrary import featureModule

testArray = np.array([0,0,0,0,1,0,0,0,1,0,0,0])

testArray = testArray.astype(bool)

print(testArray)

testArray = featureModule.createBufferedBinaryArrayFromArray(testArray, 2)

print(testArray)
