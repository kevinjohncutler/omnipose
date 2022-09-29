# Omnipose plugins

We plan to support plugins for CellProfiler and Napari. More script-focused pipelines can easily leverage a single CLI command to integrate Omnipose segmentation (for an example of this, see Teresa Lo's [fork of SuperSegger](https://github.com/tlo-bot/supersegger-omnipose)). 

## CellProfiler installation instructions

The [Windows](https://github.com/CellProfiler/CellProfiler-plugins/blob/master/Instructions/Install_environment_instructions_windows.md) instructions worked for me on Windows 11 with no issues. Follow steps 1-9. Then follow the current Omnipose GitHub installation instructions:
```
pip install git+https://github.com/kevinjohncutler/omnipose.git
```
GPU support is not baked into the environment files. See the Omnipose installation instructions to get that working. 

Here are the steps that I took to get it working on macOS:
1. Follow the [macOS](https://github.com/CellProfiler/CellProfiler-plugins/blob/master/Instructions/Install_environment_instructions_mac.pdf) instructions, steps 1-2 (Java and Conda). 
2. Clone the CellProfiler and CellProfiler-plugins repos:

    ```
    git clone https://github.com/CellProfiler/CellProfiler.git
    git clone https://github.com/CellProfiler/CellProfiler-plugins.git
    ```
3. Download this folder (or clone the entire Omnipose repo). Replace `CellProfiler/setup.py` and `CellProfiler-plugins/Instructions/cellprofiler_plugins_mac.yml` with the ones in provided in this folder. These remove several version specifications that conflict with the versions that Omnipose requires. 
4. Install the conda environment with
5. Install the local version of CellProfiler first, 
    ```    
    conda env crete -f cellprofiler_plugins_mac.yml
    ```
    then the official verison without overwriting dependencies with 
    ```
    pip install cellprofiler --no-deps --force     
    ```
    
6. Install Omnipose with 
    ```
    pip install git+https://github.com/kevinjohncutler/omnipose.git
    ```
7. Edit the CellProfiler preferences to point to the plugins path containing the RunOmnipose plugin. 

8. Relaunch CellProfiler. If the logs say that the RunOmnipose plugin cannot be loaded, try reinstalling `python-javabridge`:
    ```
    pip install python-javabridge --force
    ```
    
Time permitting, these instructions will be updated to support Apple Silicon installs and GPU acceleration. 

## Napari

A Napari plugin is on our to-do list.

