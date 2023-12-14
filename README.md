# unsuperved-kg

## Macrostrat DB explorer

The `macrostrat_db/database_explorer.ipynb` contains code to explore the database dump of the macrostrat datbase. It produces two files, a `all_columns.csv` which contains metadata
about all the columns and tables in the `macrostrat` schema and `macrostrat_graph.csv` which contains data for a graph about units metadata extracted from the query:

```
SELECT *
FROM units u
JOIN unit_liths ul
  ON u.id = ul.unit_id
JOIN liths l
  ON l.id = ul.lith_id
 -- Linking table between unit lithologies and their attributes
 -- (essentially, adjectives describing that component of the rock unit)
 -- Examples: silty mudstone, dolomitic sandstone, mottled shale, muscovite garnet granite
JOIN unit_lith_atts ula
  ON ula.unit_lith_id = ul.id
JOIN lith_atts la
  ON ula.lith_att_id = la.id
```

## REBEL based knowledge graph extraction

To extract relationships from the text corpus, we utilize the REBEL model: [https://github.com/Babelscape/rebel](https://github.com/Babelscape/rebel) which is a seq2sel model for relationship extraction.
In the `rebel_kg` directory, you can use the `kg_runner.py` to generate a knowledge graph for a text corpus. Running `python kg_runner.py --help` you can see the arguments to pass to generate the kg:
```
usage: kg_runner.py [-h] [--directory DIRECTORY] [--file FILE] [--processes PROCESSES] [--num_files NUM_FILES] --save SAVE

optional arguments:
  -h, --help            show this help message and exit
  --directory DIRECTORY
                        The directory containing the text corpus we want to process
  --file FILE           The file we want to generate the kg for
  --processes PROCESSES
                        Number of process we want running
  --num_files NUM_FILES
                        Number of files in the directory we want to save
  --save SAVE           The html file we want to save the network in
```

Alongside saving the html file, it will also save a csv file representing the knowledge graph in the same directory as the html. An example of running the command for a directory: 
```
$ python kg_runner.py --directory /ssd/dsarda/geoarchive_datasets/filtered_geoarchive_text/ --save /ssd/dsarda/geoarchive_datasets/filtered_results/temp.html --num_files 2 --processes 2`
```
This will use 2 processes to procecess 2 files from the `/ssd/dsarda/geoarchive_datasets/filtered_geoarchive_text/` directory and will save the kg network to `/ssd/dsarda/geoarchive_datasets/filtered_results/temp.html` as well as a csv file representing the kg to `/ssd/dsarda/geoarchive_datasets/filtered_results/temp.csv`

Similarily, to run for the provided example file, you can use the command:
```
$ python kg_runner.py --file example.txt --save example.html
```

The example file contains the sentence, "Jaguar is a Canadian-listed junior gold mining, development, and exploration company operating in Brazil with three gold mining complexes and a large land package covering approximately 20,000 ha." which results in a Knowledge Graph in the html file of:

![Example Knowledge Graph](images/example_graph.png)

It also produce the following csv file:
```
src,type,dst,article_id,sentence
Jaguar,country,Canadian,example,"Jaguar is a Canadian-listed junior gold mining, development, and exploration company operating in Brazil with three gold mining complexes and a large land package covering approximately 20,000 ha."
junior,subclass of,mining,example,"Jaguar is a Canadian-listed junior gold mining, development, and exploration company operating in Brazil with three gold mining complexes and a large land package covering approximately 20,000 ha."
Jaguar,product or material produced,gold,example,"Jaguar is a Canadian-listed junior gold mining, development, and exploration company operating in Brazil with three gold mining complexes and a large land package covering approximately 20,000 ha."
```

### Upload to Neo4j

To upload to Neo4j, we need to have a txt containing the following information:
```
NEO4J_URI=<uri>
NEO4J_USERNAME=<username>
NEO4J_PASSWORD=<password>
```
This information is generally provided by Neo4j. In the example below, we assume that this information is stored in `neo4j_login.txt`.

To upload the csv file produced by `kg_runner`, in this case `example.csv`, you can use the `neo4j_uploader` code as such:
```
$ python neo4j_uploader.py --login_file neo4j_login.txt --graph_file example.csv
```

## REBEL Finetuning

The problem is that REBEL model generally focuses on common terms and ignores terms that are more domain specific. For example, for the sentence:
```
Origins of dolomite in the offshore facies of the Bonneterre Formation
```

This knowledge graph is generated which completely ignores the term dolomite:
![A bad knowledge graph](images/failed_kg.jpg)


Thus, we try to finetune the REBEL model so that it recognizes these terms. The `rebel_finetuning` directory is based on the [original rebel repo](https://github.com/Babelscape/rebel). 