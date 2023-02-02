# Observatory

## Codebase and Development Environment Setup
Assume using Conda for Python package management on Linux machines. 

1. Clone this repo in your working directory (the ```--recursive``` flag is necessary to pull a dependent repo like [TURL](https://github.com/sunlab-osu/TURL) as a submodule):

    ```git clone <Observatory repo url> --recursive```
    
    ```cd observatory```

2. Create and activate the development environment:

    ```conda env create -f environment.yml ```

    ```conda activate observatory```

3. Import Observatory and TURL as editable packages to the conda environment

    ```conda develop <path to Observatory>```
    
    ```conda develop <path to TURL>```

    e.g.,
    
    ```conda develop /home/congtj/observatory```

    ```conda develop /home/congtj/observatory/observatory/models/TURL```
