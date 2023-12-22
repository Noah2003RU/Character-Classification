This project was for my Computer Imaging and Multimedia class.  The assignment was to create an algorithm that woul dtake in training data, establish the hu moments, classify them,
and use them to predict what an unknown handwritten character could be.

The first loop goes through th eknown characters, the training data, binarizes the image, and stores the hu moments in a large 2 dimensional array.  Using this, we outline the recognized characters
in red to show that they were successfully seen, and then we binarize the training data and compare its hu moments to the ones of known characters to try to get the most accurate prediction
possible.
