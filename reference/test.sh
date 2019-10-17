# installation
mvn clean package

# compile entire corpus to ast json file and create jsrc.json from all Java files under jsrc
time mvn exec:java -Dexec.mainClass=ConvertJava -Dexec.args="compilationUnit ../../../jsrc.json ../../../jsrc/"
# convert query_file.java into ast in json format
time mvn exec:java -Dexec.mainClass=ConvertJava -Dexec.args="compilationUnit query_file.json query_file.java"

# featurize asts
time python3 src/main/python/similar.py -c ../../../jsrc.json -d ../../../tmpout
# run experiments assuming that featurization has already been done
time python3 src/main/python/similar.py -d ../../../tmpout -t
# search code at index 83403 of the corpus
time python3 src/main/python/similar.py -d ../../../tmpout -i 83403
# search using the query ast in query_file.json
time python3 src/main/python/similar.py -d ../../../tmpout -f query_file.json
