# luisa-compute-rs 
Rust binding to LuisaCompute (WIP)

Inside this crate:
- An *almost* safe binding to LuisaCompute
- An EDSL for writing kernels
- A new backend implementation in pure Rust

## Table of Contents
* Example
* Usage
* Safety
## Example


## Usage 

## Safety
### API
The API is safe to a large extent. However, async operations are difficult to be completely safe without requiring users to write boilerplate. Thus, all async operations are marked unsafe. 

### Backend 
Safety checks such as OOB is generally not available for GPU backends. As it is difficult to produce meaningful debug message in event of a crash. However, the Rust backend provided in the crate contains full safety checks and is recommended for debugging.
