Following folders with data need to be in this folder.
`site` = abbreviation of the site, SMSE, PR, etc.
cwd = current working directory
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