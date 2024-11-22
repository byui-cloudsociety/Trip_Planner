# AI Expo Project and Presentation

## About

This project uses AWS SageMaker to integrate AI capabilities into our website. Our team is developing a trip planner that utilizes AI to help plan. Users will be asked questions about their trip, and the AI will compile all the information into a summary.

## How to use:

**This tutorial assumes students are using AWS Learner Labs, but will work with any instance of AWS Sagemaker**

### Setting up SageMaker

1. Log into the AWS Console and search SageMaker
2. In the navigation console on the left, select **"Notebooks"**
3. Select **"Launch Instance"**
4. Name the instance, select the type (on the lab, you are limited to _medium, large, and xlarge_), make sure the IAM role you are using is the **Lab Role**
5. Leave all the other setting at their defaults and click **"Create Notebook Instance"**

The instance takes a few minutes to launch, you can press the refresh button to update the status.

While you wait, you can:

### Download the code

1. In GitHub, click the **Code** button and copy the HTTPS URL
2. Open a Terminal in the directory or folder you want the project to be in
3. Run the command `git clone https://github.com/byui-cloudsociety/Trip_Planner.git`

OR

1. Click the **Code** button and **"Download ZIP"**
2. Select where you want the ZIP file to be saved, and wait for the download to complete
3. Unzip the file in a directory or folder where you want the project to be in

In the MS Teams chat, there is a shared file called _**poiTrainingData.csv**_, download that file somewhere you can find, we will be using it in the next step.

### Upload content to the notebook

Once the instance has started, the status should be **"Ready"** or **"InService"**

1. Select the instance, and **Open JupyterLabs**. A new tab should open
2. In the top left on the new tab, click the button that is labelled **"Upload Files"** (next to the "**+ New Launcher**" button)
3. Find the folder where you downloaded the GitHub files. Open "notebook". Select and upload the following files:
   - sage_tfidf.py
   - sage_word2Vec.py
   - launch.ipynb
4. Find the folder where you downloaded the training data CSV file. Select and upload that file
5. In total **you should have _4_ files**

### Running the Code

Every time you launch or re-launch the instance, you will have to train the model with the data.

1. Double-click **"launch.ipynb"** from the JupyterLabs file browser
2. Click **"Run"**, and **"Run all Cells"**

When the code process gets to the 5th cell, there will be some user interaction.

#### After the initial training, you can select the 5th cell and run it alone.

## Next Steps

This code is available for anyone to use and change, locally.

If you are part of the BYU-I Cloud Computing Team, talk to a member of the presidency about how you can upload changes for the whole project.

**There is a way to connect the frontend part of the website to the model in SageMaker, but that is for a future project.**

## Video Presentation

The video presentation made for the BYU-I AI Expo will go here
