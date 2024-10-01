# Palm 
Contains the predictions of male, female flowers & the absence of flowers using two methods: Random Forest & Deep Learning (mainly CNNs).

## Info Paths
### 1. Data path

Following folders with data need to be present. 

cwd = current working directory
|___
    `NeuralNetwork_Testing`
    |___
        |
        |___`dataCIGE` = folder which holds the data of the different sites
        |
        |___`data_{site}` = folder which holds the data of site X
            |
            |___`Events_tree_{site}_Charge.npy` = file for tree events (mandatory)
            |___`dfMeteoInfo.npy` = file for meteorological events

total path =  cwd + `dataCIGE/data_{site}/Events_tree_{site}_Charge.npy`
`site` = abbreviation of the site, SMSE, PR, etc.
Paths may be sometimes absolute and need to be set new if the repository is used on a different device.
