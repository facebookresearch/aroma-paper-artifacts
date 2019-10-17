# Paper Artifacts for "Aroma: Code Recommendation via Structural Code Search"

This archive contains source code and data to help reproduce the evaluation results in the paper. This document explains the format and content of each file.

## Datasets

### The GitHub Code Corpus

File: `datasets/github_repositories.csv`

This file contains information to recreate the code corpus for evaluation as described in Section 4 of the paper. It contains a list of open-source repositories on GitHub where Java is the main programming language and Android is the project topic. For each repository, we provide its full name, the SHA-1 hash of the git commit which we checked out in our evaluation, and the date of that commit. We also provide a web URL which links to the GitHub repository page on the particular commit in the dataset.

We verified all of the repositories and commits in this file are publicly available as of July 31, 2019. The list contains 5,339 repositories. The rest of the repositories from the original list of 5,417 repositories were either moved, deleted, or became private. Our company policy forbids us to share the source code of these repositories directly in this archive. However, we believe since the majority of the code corpus can still be retrieved, the evaluation results should not be significantly different from the original evaluation.

Note that for legal reasons, we cannot directly share the source code in these repositories. However, as approved by the program chair, we instead provide the URL that links to the particular version of the GitHub repository used in our evaluation dataset.

The format of this file is ASCII-encoded CSV (Comma-separated values). The first row is the table header, and each row starting from row 2 contains information of one repository.

### The Stack Overflow Code Snippets

File: `datasets/stack_overflow_code_snippets.csv`

This file contains the code snippets used in the evaluation of code recommendation quality in Section 4 of the paper. It contains 64 code snippets from Stack Overflow. We retrieved these code from the [Stack Exchange Data Dump](https://archive.org/details/stackexchange), sharable under the [CC-BY-SA 3.0 Creative Commons License](https://creativecommons.org/licenses/by-sa/3.0/). All the web URL links are publicly available as of July 31, 2019.

The format of this file is ASCII-encoded CSV (Comma-separated values). The first row is the table header, and each row starting from row 2 contains information of one code snippet. The column descriptions are as follows:

- `question_url`: a web URL that links to the question post on Stack Overflow.
- `question_author`: the screen name of the author of the question post on Stack Overflow.
- `question_author_url`: a web URL that links to the question author's user profile page.
- `question_upvotes`: the number of upvotes of the question post, a measure of the popularity of the topic.
- `question_title`: the full title of the question post on Stack Overflow.
- `answer_url`: a web URL that directly links to the answer post and the corresponding code snippets on Stack Overflow.
- `answer_author`:  the screen name of the author of the answer post on Stack Overflow.
- `answer_author_url`: a web URL that links to the answer author's user profile page.
- `answer_upvotes`: the number of upvotes of the answer post, a measure of the popularity of the proposed solution.
- `full_code_snippet`: the full code snippet extracted from the answer post, unmodified.
- ` query_code_snippet`: the modified code snippet we used to assess the code recommendation quality on *full* code snippets. The modification typically only involves stripping the class or method declarations and retaining the statements.
- `categorization`: the categorization of the code recommendation result, corresponding to the categories as described in Section 4.3 of the paper.
- `partial_code_snippet`: the modified, partial code snippet we used to assess the code recommendation quality on *partial* code snippets. In principle, we picked the first half of the statements in the full code snippet that are semantically meaningful. For the full code snippets with only one line, we were not able to create a "partial query", and thus this column remains blank.
- `partial_code_categorization`: the categorization of the code recommendation result, as described in Section 4.2 of the paper.

Note that for legal reasons, we cannot directly share the code recommendation results synthesized from the open-source GitHub code corpus. Examples of each category of code recommendation are provided in Table 1 of the paper.

### The Code Snippets from the BigGroum Paper (Mover et al.)

File: `datasets/biggroum_code_snippets.csv`

This file contains the code snippets used in the comparison with pattern-oriented code completion in Section 4.4 of the paper. It contains 15 coding patterns in the paper artifacts of the BigGroum paper (Mover et al.). The original paper artifacts are publicly available at https://goo.gl/r1VAgc, as of July 31, 2019.

The format of this file is ASCII-encoded CSV (Comma-separated values). The first row is the table header, and each row starting from row 2 contains information of one code snippet. The column descriptions are as follows:

- `title`: the title of the pattern in the Mover et al. dataset.
- `code_snippet`: the full code snippet representing the coding pattern in the Mover et al. dataset, unmodified.
- `query_code_snippet`: the modified code snippet we used to assess the code recommendation quality on *partial* code snippets. In principle, we picked the first half of the statements in the full code snippet that are semantically meaningful.
- `recommendation_result`: "Recall" means in one of the code recommendation results given the query code snippet, we found the original coding pattern as described in the Mover et al. dataset. "No Recall" means we did not retrieve the original coding pattern.
- `recommendation_code_snippet_url`: a web URL that links to the code snippet used as the base snippet for creating the recommended code snippet. For legal reasons we cannot directly share the code recommendation results synthesized from the open-source GitHub code corpus. However, as approved by the program chair, we instead provide links to the specific line ranges in the specific versions of the files on GitHub from which the code recommendation results are synthesized. The URLs are publicly accessible as of July 31, 2019.



## Reference Implementation and Examples

### Java to Simplified Parse Tree Converter

File: `reference/src/main/java/ConvertJava.java`

This file contains a reference implementation of a parser that parses Java source files, and extracts Simplified Parse Trees, as defined in Section 3.1 of the paper.

An example input file is `reference/data/example_query.java`. The example output by the parser program is `reference/data/example_query.json`. The output file contains information about the input, as well as the Simplified Parse Tree in JSON format.

### Main Algorithm

File: `reference/src/main/python/similar.py`

This file contains a reference implementation of the main Aroma algorithm. It contains the indexing stage, as well as the search stages (light-weight search, prune and rerank, cluster and intersect).

Specifically, the features produced for the same example input file is listed in `reference/data/example_features.txt`, where each line represents a feature.

### Build and Run

File: `reference/test.sh`

To build and run the reference implementation, follow the instructions in the `test.sh` file. Java, Maven and Python must be available in the environment.

The other files in the `reference/` directory are support files (Maven build configurations and ANTLR4 grammar files).

## References

- Sergio Mover, Rhys Olsen, Bor-Yuh Evan Chang, Sriram Sankaranarayanan. _Mining Framework Usage Graphs from App Corpora_. SANER 2018.