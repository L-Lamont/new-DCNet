# new-DCNet
A modified version of slee5777/DCNet so training could be distributed using fastai
- _Requires fastai<=2.6.3, breaking changes to distribution introduced in 2.7_

Uses segment anything model in notebooks in sam subdirectory to generate masks
using the center point locations, currently only example tests to prove concept
