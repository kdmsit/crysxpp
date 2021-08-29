# CrysXPP: An Explainable Property Predictor for Crystalline Materials (NPJ Computational Submission)

This is software package for Crsytal Explainable Property Predictor(CrysXPP) that takes as input
any arbitary crystal structure in .cif file format and predict different state and elastic properties
of the material.


##  Requirements

## Usage

### Define a customized dataset 
(Customiation is adopted From CGCNN Paper)

To input crystal structures to CrysAE and CrysXPP, you will need to define a customized dataset. Note that this is required for both training and predicting. 

Before defining a customized dataset, you will need:

- [CIF](https://en.wikipedia.org/wiki/Crystallographic_Information_File) files recording the structure of the crystals that you are interested in
- The target properties for each crystal (not needed for predicting, but you need to put some random numbers in `id_prop.csv`)

You can create a customized dataset by creating a directory `root_dir` with the following files: 

1. `id_prop.csv`: a [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) file with two columns. The first column recodes a unique `ID` for each crystal, and the second column recodes the value of target property. 
2. `atom_init.json`: a [JSON](https://en.wikipedia.org/wiki/JSON) file that stores the initialization vector for each element. An example of `atom_init.json` is `data/sample-regression/atom_init.json`, which should be good for most applications.

3. `ID.cif`: a [CIF](https://en.wikipedia.org/wiki/Crystallographic_Information_File) file that recodes the crystal structure, where `ID` is the unique `ID` for the crystal.

The structure of the `root_dir` should be:

```
root_dir
├── id_prop.csv
├── atom_init.json
├── id0.cif
├── id1.cif
├── ...
```

There is a examples of customized dataset in the repository: `../data/`, where in id_prop file we have formation energy values.
You can use the utils.py file to generate the data as per this format.

### Train a CrysAE model :

Before training a new CrysAE model, you will need to:

- [Define a customized dataset](#define-a-customized-dataset) at `root_dir` to store the structure-property relations of interest.

```bash
python main.py data-path = '../data/' --is-global-loss = <1/0> --is-local-loss = <1/0>  --save-path = <path_to_save_pretrained_model>
```
Once the training is done the saved model will be saved at save-path.

### Train a CrysXPP model :
Before training a new CrysXPP model, you will need to:

- [Define a customized dataset](#define-a-customized-dataset) at `root_dir` to store the structure-property relations of interest.

You can train the property predictor module by the following command :

```bash
python prop.py --pretrained-model=<Pretrain_CrysAE_path> --batch-size=512 --epoch=200 --test-ratio=0.8
```
Here you can set set the following hyperparameters :

- lrate : Learning Rate (Default : 0.003).
- atom-feat : Atom Feature Dimension (Default : 64).
- nconv : Number of Convolution Layers (Default : 3).
- epoch : Number of Training Epochs (Default : 200)
- batch-size : Batch size of data (Default : 512).


After training, you will get following files :

- ``../model/model_pp.pth`` : Saved model for that particular property.
-  ``../results/Prediction/<DATE>/<DATETIME>/out.txt`` : All the traing results for all epochs and all the hyperparameters are saved here.

## License

CrysXPP is released under the MIT License.