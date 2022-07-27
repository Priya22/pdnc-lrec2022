# The Project Dialogism Novel Corpus

This repository contains data and code associated with the LREC 2022 paper [The Project Dialogism Novel Corpus:
A Dataset for Quotation Attribution in Literary Texts](https://arxiv.org/abs/2204.05836).

**Note**: The official repository for the Project Dialogism Novel Corpus has been moved to [here](https://github.com/Priya22/project-dialogism-novel-corpus), and will be updated with new novels as they are annotated.

## Data and Annotation
The PDN Corpus contains annotations for speaker, addressees, referring expression, and pronominal mentions for all quotations in 22 novels. The list of novels can be found in the file `ListOfNovels.txt`.

In the `data` folder, for each novel, there are three files:
- `text.txt`: The text of the novel
- `quotations.csv`: This is a CSV file where each row contains, for a quotation:
    - The text of the quotation
    - The corresponding character-byte spans from the novel text
    - The name of the speaker
    - The names of the addressees
    - Texts of the mentions annotated within the quotation
    - Character-byte spans of the mentions from the novel text
    - The entities referred to by the above mentions
    - The type of the quotation (implicit, anaphoric, or explict)
    - The referring expression associated with the quotation, if any

- `charDict.pkl`: Each character-entity in a novel is assigned a unique ID. This pickle file is a dictionary with the following key-value pairs:
    - id2names: The list of names (aliases) associated with each ID
    - name2id: A reverse-mapping of each character alias to the corresponding ID
    - id2parent: The *main* character name associated with each ID

### Helper File
The IPython notebook `load_data.ipynb` shows how to load and read the data files for a novel. 
### Annotation Guidelines
The full text of the annotation guidelines that were used to annotate this corpus can be viewed at [this link](https://docs.google.com/document/d/1eBsX2rjdLBkmA-kWB_jHCxC1nmbzinH04WUg9PeN_2A/edit?usp=sharing).
## Code
The `code` folder contains scripts needed to run the semi-supervised classification approach described in Section 5.1.3 of the paper.
You can run the classifier for a novel with the following command:

        python semi_sup_clf.py --novel <novel-name> --save_path outputs/

where `<novel-name>` should be substituted with the corresponding folder name in the `data` folder for a novel. 

## Authors
Please contact the authors of the paper with any queries:
- [Krishnapriya Vishnubhotla](https://priya22.github.io/) (University of Toronto)
- [Adam Hammond](https://www.adamhammond.com/) (University of Toronto)
- [Graeme Hirst](https://www.cs.toronto.edu/~gh/) (University of Toronto)

Contact: vkpriya@cs.toronto.edu, adam.hammond@utoronto.ca, gh@cs.toronto.edu

