###

# Exercises: detach(), item() and requires_grad
# Predict the output before you run each line.
#
# 1. requires_grad
#    - a = torch.tensor([1.0, 2.0]); print(a.requires_grad)
#    - b = a * 2; print(b.requires_grad)
#    - Set a to track gradients with requires_grad_(), recompute b. Does b change?
#    Question: why does b track gradients when you never asked it to?
#
# 2. grad_fn
#    - print(b.grad_fn); c = b.sum(); print(c.grad_fn)
#    Question: what does each grad_fn record, and in what order?
#
# 3. detach() breaks the graph
#    - d = b.detach(); print(d.requires_grad, d.grad_fn)
#    - d[0] = 100; print(b). Does b change?
#    Question: does detach() copy the data or share it with the original?
#
# 4. item() returns a Python number
#    - print(c.item()); print(type(c.item())); print(type(c))
#    - Call .item() on a tensor with two elements. What error, and why?
#    Question: why does item() only work on single values?
#
# 5. Why the checksum uses detach() and item()
#    - Sum p.sum() over the parameters of nn.Linear(3, 2), without detach().
#      Does the result have requires_grad set?
#    - Repeat with p.detach().sum().item(). What type do you get?
#    Question: what would break if you logged the sum without item()?
#
# 6. no_grad() versus detach()
#    - Inside `with torch.no_grad():` compute e = a * 3; print(e.requires_grad)
#    - Repeat with detach(). Any difference?
#    Question: when would you pick one over the other?
#
# Reading:
# - https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
# - https://pytorch.org/docs/stable/notes/autograd.html
# - https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html
# - https://pytorch.org/docs/stable/generated/torch.Tensor.item.html