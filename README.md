# Observatory

## Codebase and Development Environment Setup
Assume using Conda for Python package management on Linux machines. 

1. Clone this repo in your working directory:

    ```git clone <Observatory repo url>```
    
    ```cd observatory```

2. Create and activate the development environment:

    ```conda env create -f environment.yml ```

    ```conda activate observatory```

2. Add [TURL](https://github.com/sunlab-osu/TURL) as a submodule

    ```git submodule add <TURL repo url> observatory/models/TURL```

3. Import Observatory and TURL as editable packages to the conda environment

    ```conda develop <path to Observatory>```
    
    ```conda develop <path to TURL>```

    e.g.,
    
    ```conda develop /home/congtj/observatory```

    ```conda develop /home/congtj/observatory/observatory/models/TURL```
