# Control unit
To ensure that there is a clear separation between the different parts of the
control unit. Each part is treated as its own sub-component and they are only
allowed to include headers from the include folder in other sub-components. The
model and tests are an exception to this as they will never be shared and will
not be built as a standalone component.

## Folder structure
* privileged - Privileged control unit code for the hardware
* unprivileged - Unprivileged control unit code used by the privileged part and the model
* model - Code for running the unprivileged part in the Model
* common - Utility code used by all the parts
* include - Headers for external components
* tests - Tests for all the parts
