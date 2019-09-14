import numpy as np

import model
descriptor = model.create_model()

img1_representation = descriptor.predict(model.preprocessing_image('images/1.png'))[0, :]
img2_representation = descriptor.predict(model.preprocessing_image('images/2.png'))[0, :]

np.save("descriptors/img1_descriptor", img1_representation)
np.save("descriptors/img2_descriptor", img2_representation)

print("saved")
