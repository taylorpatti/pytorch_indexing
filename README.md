# pytorch_indexing

Functions for efficient, large-scale, elementwise tensor-tensor comparisons with PyTorch Autograd support.

The "compare_all_elements" function takes two Pytorch tensors and returns all of the indices from each where they share a common element.

def compare_all_elements(tensorA, tensorB, max_val, data_split=1):
    """
    Description.....
    
    Parameters:
        tensorA:         first array to be compared (1D torch.tensor of ints)
        tensorB:         second array to be compared (1D torch.tensor of ints)
        max_val:         the largest element in either tensorA or tensorB (real number)
        data_split:      the number of subsets to split the mask up into (int)
    Returns:
        compared_indsA:  indices of tensorA that match elements in tensorB (1D torch.tensor of ints, type torch.long)
        

For example, in order to find the indices of matching values (as used in the torch_sparse sparse-sparse matrix multiplication function "spspmm"):

input:
tensorA = torch.tensor([0, 1, 2, 3, 2])
tensorB = torch.tensor([2, 3, 0, 0])
indsA, indsB = pytorch_indexing.compare_all_elements(tensorA, tensorB, 3)

output:
tensor([2, 4, 3, 0, 0])
tensor([0, 0, 1, 2, 3])


Alternatively, we can transform our tensors in order to get the indices of elements that together uphold a different condtion. For example: tensorA + tensorB == 10

input:
tensorA = torch.tensor([0, 20, 15, 17, 5])
tensorB = torch.tensor([-5, -10, 4, -3])
#solve tensorA + tensorB == 10 --> tensorA - 10 == -tensorB, now we can test for equality with
#temporary tensors tempA and tempB
tempA = tensorA - 10
tempB = -tensorB
indsA, indsB = pytorch_indexing.compare_all_elements(tempA, tempB, 10)

output:
tensor([2, 1])
tensor([0, 1])

