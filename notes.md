# PyTorch
- PyTorch tensors are very similar to NumPy n-dimensional arrays
- torch package = top level package + tensor library
- torch.nn + torch.autograd = primary workhorse packages
- programming w pytorch, its very very close to making neural netwoks from scratch!

## Philosphy of PyTorch 
- Stay out of the way
- Cater to the impatient
- Promote linear code-flow
- Full interop w Python
- Be ASAP


## CUDA and Why Deep Learning Requires GPUs
- GPUs are very good at specialized computations
    - Computations are done in parallel
    - Parallel computing -> smaller computations are done parallel manner and synchronized
    - Number of smaller computations = Number of cores available
- Neural networks are parallel computations -> there is very little effort required to break a neural network into parallel computations
- CUDA is the software layer that provides the API layer for the hardware
- GPU is only faster for specialized operations

## Tensors Explained
- Number, Array, 2D-Array : Computer Science Terms
- Scalar, Vector, Matrix : Mathematics
- terms respectively correspond
- When +n indices are required, we just use nd-tensor to refer to nd-array
- Tensors are multi-dimensional arrays


## Rank, Axes, and Shape
### Rank
- Rank of a tensor = number of dimensions present within the tensor
    - A rank 2 tensor includes:
        - matrix, 2d-array, 2d-tensor
    - How many indices do you need to access an element
    - Length of shape = Rank (think of it as number of columns)
### Axes
- Axes of a tensor = specific dimension of a tensor
    - A tensor has two 2 ranks = It has 2 axes
    - Length of each axis = how many indices we have to available
    - We "run" along an axis.

### Shape
- Shape of a tensor = determined by the length of each axis.
    - How many indices along each axis.

### Process of Reshaping
- Shape changes the grouping of the terms but not the underlying terms themselves


## CNN Tensor Shape
- Last two axes in a CNN input shape are height and weight
- second axis: coloring of the CNN
- First axis: Batch size
    - Batches of sample instead of single sample
    - How many batches are in our image
- CNN changes the output and shape of our tensors

## PyTorch Tensors
- Data structures used when we program in PyTorch
- Tranform data into our tensors
- PyTorch Tensor has a dtype
    - data types for the tensor
    - Tensors also have specific devices that they run on
    - layout means how data is laid out in memory
- Tensors contain data of a uniform type (dtype)
- Tensor computations between tensors depend on dtype and device.

## Tensor Operation Types

### Reshaping
- Most important tensor operation
- Squeezing a tensor = removes dimensions/azes that have a length of one
- Unsqueezing a tensor adds a dimension with a length of one
- Flattenning a tensor = reshapes tensor to have shape that is equal to the number of elements contained in the tensor
    - Remove all of the dimensions except for one
    - Make it a rank One tensor


## Broadcasting and Element-Wise Operations
- Element-wise operation: operation between two tensors that operates on corresponding elements within the respective tensors
    - same positions within the tensors
- Two tensors must have the SAME SHAPE for element-wise operations to occur
- Arithmetic operations using scalar values
    - Tensor broadcasting = how tensors of different shapes are treated
    - t1 + 2  => scalar tensor of 2 is being broadcasted to the t1s shape
    - the scalar tensor gets blown/minimized down to a tensor's shape and then we can do element-wise operations
- Comparison operators between two tensors:
    - 0 if operation is False
    - 1 if operation is True
- Elementwise = Componentwise = Pointwise

## Argmax and Reduction Tensor Ops
- A reduction operation on a tensor is an operation that reduces the num of elements contained in the tensor
- A sum operation is a REDUCTION operation as it reduces the number of elements to a single element
- Argmax
    - which index location give us the maximum value in a tensor
    - tensor is reduced to a tensor which indicates where the max value is

# Why Study A Dataset
- Who created the dataset
- How was it created
- What transformations were used
- What intent does dataset have
- Possilble consequences
- Fashion MNIST Dataset
    - has 10 classes (intentional to imitate MNIST)
- ETL Process
    - Extract the data
    - Transform the data
    - Load the data
- Dataset Class
    - must implement __len__ method and __getitem__ method
    - abstract class that we can use to reprsent our dataset so that we pass to PyTorch models
    - Class imbalance is a common problem
- Neural Network in PyTorch
    - Extend the nn.Module base class
    - Define layers as class attributes
    - Implement the forward() method
-For each layer
    - Weight tensor + forward function definition
    - Parameters as place holders
    - Arguments are what we pass to the function
    - Parameters are like local variables and arguments are what give values to parameters


#### Two Types of Parameters
- Hyperparameters
    - values are chosen manually and arbitrarily
    - based on trial and error 
    - utilize values that have proven to work well in the past
    - Kernel_size => sets the filter size
        - convolutional kernel/filter
        - size to perform the convolution operation
        - produces an output_channel
        - setting the number of filters
        - out_channels are also called feature maps
            - in linear layers, out_channels are called out_features
    - Trend is in CNN, increase out_channels as layers increase
        - When you get to linear layers, decrease out_channels to narrow down
- Data dependent hyperparameters
    - whose values depend on the data
    - the in_channels and in_features are data dependent hyper parameters
    - When we switch from conv layers to linear layers, we have to flatten


## Learnable Parameters
- Learnable parameters are parameters whose values are learned during the training process.
    - We start out with a set of values and those values are updated in an iterative fashion as the network learns
    - When a network learns, we mean that the network is learning the correct values for the learnable parameter
- __repr__() is a method built for official string representation for an obejct
- stride tells the conv layer how far the kernel_layer should slide
- All filters are represented using a single tensor
- Filters have a depth that accoutns for the color channels
- Weight matrix is multiplied w the input layer which results in a diminished output layer