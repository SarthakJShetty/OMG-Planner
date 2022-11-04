# OMG-Planner 
[[webpage](https://liruiw.github.io/planning.html), [paper](https://arxiv.org/abs/1911.10280)]


![image](assets/top.PNG)

### NGDF OMG installation

* First set up ngdf according to the instructions in the README

* `git clone https://github.com/liruiw/OMG-Planner.git --recursive`
* `pip install -r requirements.txt`

* Install [ycb_render](ycb_render)  

    ```Shell
    cd ycb_render
    python setup.py develop
    ```

* Install Eigen from the Github source code [here](https://github.com/eigenteam/eigen-git-mirror)
    ```
    git clone https://github.com/eigenteam/eigen-git-mirror
    cd eigen-git-mirror
    mkdir build
    cd build
    cmake ..
    sudo make install
    ```

* Install the submodule Sophus. Check if the submodule is correctly downloaded.

    ```Shell
    cd Sophus
    mkdir build
    cd build
    cmake ..
    make -j8
    sudo make install
    ```

* Install PyTorch (if not already installed from the ngdf instructions)
    ```
    pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
    ```

* Compile the new layers under layers we introduce.
    ```Shell
    cd layers
    python setup.py install
    ```

* Install the submodule PyKDL. Check this tutorial [here](https://git.ias.informatik.tu-darmstadt.de/lutter/ias_pykdl/blob/8b864ccf81763439ba5d45a359e1993208c2247c/pykdl.md) if there is any issue with installing PyKDL.

    ```bash
    conda install sip==4.19.8
    cd orocos_kinematics_dynamics
     
    export ROS_PYTHON_VERSION=3
    cd orocos_kdl
    mkdir build; cd build;
    cmake ..
    make -j8; sudo make install
      
    cd ../../python_orocos_kdl
    mkdir build; cd build;
    cmake ..  -DPYTHON_VERSION=3.7.9 -DPYTHON_EXECUTABLE=$CONDA_PREFIX/bin/python3.7
    make -j8;  cp PyKDL.so $CONDA_PREFIX/lib/python3.7/site-packages/
    ```

* pip install base OMG-Planner module
    ```
    cd OMG-Planner
    pip install -e .
    ```

* Run ```./download_data.sh``` for data (Around 600 MB).

* Now you can run the pybullet scene
    ```
    python -m bullet.panda_scene --method origOMG_known --eval_type 1obj_float_fixedpose_nograv --dset_root=/home/thomasweng/data/manifolds/acronym_mini_relabel -o=/home/thomasweng/data/manifolds/pybullet_eval/dbg
    ```
    or 
    ```
    python eval_scripts/run_eval.py --exp_name=dbg --data_root=/home/thomasweng/data/manifolds --trials=1 --render
    ```

* If processing new object models, you'll need to install additional packages
    * SDFGen
    * blenderpy https://blender.stackexchange.com/questions/126959/how-to-build-blender-as-python-module
    

