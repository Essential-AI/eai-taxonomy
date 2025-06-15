# Essential-Web v1.0: Dataset Curation Tools

This repository contains annotation tools and dataset filters from the **Essential-Web v1.0** project. 

**Dataset**: [EssentialAI/essential-web-v1.0](https://huggingface.co/datasets/EssentialAI/essential-web-v1.0)

## Repository Contents

### Annotation System
- **`src/eai_taxonomy/annotation/`**: Document classification pipeline
  - `prompts/`: System and task prompts for EAI-Distill-0.5b
  - `annotation.py`: Core annotation logic
  - `annotation_examples/`: Sample outputs

### Dataset Filters
- **`src/eai_taxonomy/notebooks/`**: Spark filters for domain-specific datasets
  - `eai_taxonomy_top_math.ipynb`: Math dataset curation
  - `eai_taxonomy_stem_w_dclm.ipynb`: STEM dataset filters
  - `eai_taxonomy_med_w_dclm.ipynb`: Medical dataset filters

## Installation

```bash
pip install -r requirements.txt
# OR
uv sync
```

## Performance

Datasets curated using these filters demonstrated competitive performance:
* 🧮 **Math**: within 8.0% of web-curated baselines
* 💻 **Web Code**: 14.3% above web-curated baselines
* 🔬 **STEM**: 24.5% above web-curated baselines
* 🩺 **Medical**: 8.6% above web-curated baselines

# Dataset Schema Documentation

## Overview

This dataset contains web-crawled text data with comprehensive metadata, quality signals, and taxonomic classifications. Each record represents a document extracted from web archives with detailed provenance tracking and quality assessment metrics.

## Core Fields

| Field | Type | Description | Path |
|-------|------|-------------|------|
| `id` | `Int64` | Unique identifier based on document hash | `id` |
| `text` | `String` | The main textual content of the document | `text` |

## EAI Taxonomy Classification

Comprehensive hierarchical classification system with primary and secondary labels - the most important feature of this dataset. The taxonomy is designed to provide detailed subject categorization, document type identification, content quality assessment, and extraction quality indicators.

<details>
<summary><strong>Free Decimal Correspondence (FDC)</strong></summary>

A Dewey Decimal-inspired classification system with 3-level hierarchical labels. The FDC provides nested categories where each successive level refines its parent category. It's designed to be compatible with the Dewey Decimal System for library cataloging.

**Level Structure:**
- **Level 1**: Top-level categories (0-9) covering broad subject areas like General works, Philosophy, Religion, Social Sciences, etc.
- **Level 2**: Sub-divisions (00-99) that refine Level 1 categories
- **Level 3**: Specific categories (000-999) that further refine Level 2 categories

| Component | Description | Path |
|-----------|-------------|------|
| Primary Code | Main classification code | `eai_taxonomy.free_decimal_correspondence.primary.code` |
| Primary Level 1 | Top-level category (0=General works, 1=Philosophy, 2=Religion, 3=Social Sciences, 4=Language, 5=Science, 6=Technology, 7=Arts, 8=Literature, 9=History/Geography) | `eai_taxonomy.free_decimal_correspondence.primary.labels.level_1` |
| Primary Level 2 | Mid-level category | `eai_taxonomy.free_decimal_correspondence.primary.labels.level_2` |
| Primary Level 3 | Specific category | `eai_taxonomy.free_decimal_correspondence.primary.labels.level_3` |
| Secondary Code | Alternative classification code | `eai_taxonomy.free_decimal_correspondence.secondary.code` |
| Secondary Level 1 | Alternative top-level category | `eai_taxonomy.free_decimal_correspondence.secondary.labels.level_1` |
| Secondary Level 2 | Alternative mid-level category | `eai_taxonomy.free_decimal_correspondence.secondary.labels.level_2` |
| Secondary Level 3 | Alternative specific category | `eai_taxonomy.free_decimal_correspondence.secondary.labels.level_3` |

We recommend this viewer for easily navigating the FDC categories when curating filters: https://www.librarything.com/mds

</details>

<details>
<summary><strong>Bloom's Taxonomy Integration</strong></summary>

Based on Anderson and Krathwohl's 2001 revision of Bloom's Taxonomy of Educational Objectives, providing two complementary categorization dimensions for educational content analysis.

### Knowledge Domain
Categorizes the type of knowledge demonstrated in the document:

| Component | Description | Path |
|-----------|-------------|------|
| Primary Code | Main knowledge domain code | `eai_taxonomy.bloom_knowledge_domain.primary.code` |
| Primary Label | Main knowledge domain label | `eai_taxonomy.bloom_knowledge_domain.primary.label` |
| Secondary Code | Alternative knowledge domain code | `eai_taxonomy.bloom_knowledge_domain.secondary.code` |
| Secondary Label | Alternative knowledge domain label | `eai_taxonomy.bloom_knowledge_domain.secondary.label` |

**Possible Values:**
| Code | Label | Description |
|------|-------|-------------|
| `-1` | Abstain | Unable to determine |
| `1` | Factual | Basic elements to learn or solve problems |
| `2` | Conceptual | Interrelationships between basic elements within larger context |
| `3` | Procedural | Methods and techniques in the discipline |
| `4` | Metacognitive | Awareness of how learning works in relation to oneself |

### Cognitive Processing Level
Assesses the learning and thinking skill levels demonstrated by the document author:

| Component | Description | Path |
|-----------|-------------|------|
| Primary Code | Main cognitive process code | `eai_taxonomy.bloom_cognitive_process.primary.code` |
| Primary Label | Main cognitive process label | `eai_taxonomy.bloom_cognitive_process.primary.label` |
| Secondary Code | Alternative cognitive process code | `eai_taxonomy.bloom_cognitive_process.secondary.code` |
| Secondary Label | Alternative cognitive process label | `eai_taxonomy.bloom_cognitive_process.secondary.label` |

**Possible Values:**
| Code | Label | Description |
|------|-------|-------------|
| `-1` | Abstain | Unable to determine |
| `1` | Remember | Retrieve relevant knowledge from memory |
| `2` | Understand | Determine meaning of instructional messages |
| `3` | Apply | Use a procedure in a given situation |
| `4` | Analyze | Break materials into components and determine relationships |
| `5` | Evaluate | Make judgments based on criteria and standards |
| `6` | Create | Create new or original work |

</details>

<details>
<summary><strong>Document Characteristics</strong></summary>

### Document Type v1
In-house classification of common web document types and formats:

| Component | Description | Path |
|-----------|-------------|------|
| Primary Code | Main document type code | `eai_taxonomy.document_type_v1.primary.code` |
| Primary Label | Main document type label | `eai_taxonomy.document_type_v1.primary.label` |
| Secondary Code | Alternative document type code | `eai_taxonomy.document_type_v1.secondary.code` |
| Secondary Label | Alternative document type label | `eai_taxonomy.document_type_v1.secondary.label` |

**Possible Values:**
| Code | Label | Examples |
|------|-------|----------|
| `-1` | Abstain | Unable to classify |
| `1` | News/Editorial | CNN articles, opinion columns |
| `2` | Academic/Research | ArXiv papers, research articles |
| `3` | Reference/Encyclopedic/Educational | FAQs, Wikipedia entries |
| `4` | Code/Software | GitHub repos, code examples |
| `5` | Social/Forum | Conversation threads, Q&A boards |
| `6` | Promotional/Advertisement | Product pages, calls to action |
| `7` | Search/Directory/Bibliography | Link pages, search results |
| `8` | Adult/Pornographic | Adult content |
| `9` | Personal/Misc | Blogs, user profiles |
| `10` | Machine-Generated | Lorem ipsum, garbled text |
| `11` | Legal/Regulatory | Contracts, terms of service |
| `12` | Government/Political | Legislation, press releases |
| `13` | Literary/Creative | Poems, short stories |
| `14` | Reviews/Critiques | Film critiques, product reviews |
| `15` | E-Commerce/Marketplace | eBay listings, Amazon pages |
| `16` | Images/Videos/Audio | YouTube videos, Imgur pages |
| `17` | Other/Unclassified | Documents that resist classification |

### Document Type v2
Updated classification based on WebOrganizer taxonomy with refined categories for improved document classification accuracy:

| Component | Description | Path |
|-----------|-------------|------|
| Primary Code | Main document type code (v2) | `eai_taxonomy.document_type_v2.primary.code` |
| Primary Label | Main document type label (v2) | `eai_taxonomy.document_type_v2.primary.label` |
| Secondary Code | Alternative document type code (v2) | `eai_taxonomy.document_type_v2.secondary.code` |
| Secondary Label | Alternative document type label (v2) | `eai_taxonomy.document_type_v2.secondary.label` |

**Complete Value Mapping:**
| Code | Label | Examples |
|------|-------|----------|
| `-1` | Abstain | Documents requiring human review |
| `1` | About (Org.) | Company about pages, mission statements |
| `2` | About (Personal) | Personal bios, LinkedIn profiles |
| `3` | Academic Writing | Research papers, abstracts, dissertations |
| `4` | Audio Transcript | Interview transcripts, court records, captions |
| `5` | Comment Section | Reddit threads, blog comments |
| `6` | Content Listing | Site maps, product catalogs, directory listings |
| `7` | Creative Writing | Song lyrics, novel excerpts, poetry |
| `8` | Documentation | API docs, README files, user manuals |
| `9` | FAQ | FAQ pages, Q&A lists |
| `10` | Knowledge Article | Wikipedia articles, Britannica entries |
| `11` | Legal Notices | Privacy policies, license agreements, terms of service |
| `12` | Listicle | Buzzfeed-style articles, "Top 10" lists |
| `13` | News (Org.) | Government blog posts, corporate announcements |
| `14` | News Article | Newspaper articles, CNN content, breaking news |
| `15` | Nonfiction Writing | Editorials, obituaries, memoirs, opinion pieces |
| `16` | Personal Blog | Personal journals, diary entries, lifestyle blogs |
| `17` | Product Page | Product descriptions, course offerings, sales pages |
| `18` | Q&A Forum | Quora posts, Stack Exchange discussions |
| `19` | Spam / Ads | SEO keyword stuffing, promotional spam |
| `20` | Structured Data | Datasheets, glossaries, JSON files, databases |
| `21` | Customer Support | Help articles, troubleshooting guides |
| `22` | Truncated | Paywalled sites, image galleries, partial content |
| `23` | Tutorial | Cooking recipes, WikiHow pages, step-by-step guides |
| `24` | User Review | Yelp reviews, TripAdvisor feedback, product reviews |
| `25` | Other/Unclassified | Miscellaneous documents not fitting other categories |

### Extraction Artifacts
Assessment of technical extraction quality, identifying issues from HTML-to-text conversion:

| Component | Description | Path |
|-----------|-------------|------|
| Primary Code | Main extraction artifact code | `eai_taxonomy.extraction_artifacts.primary.code` |
| Primary Label | Main extraction artifact label | `eai_taxonomy.extraction_artifacts.primary.label` |
| Secondary Code | Alternative extraction artifact code | `eai_taxonomy.extraction_artifacts.secondary.code` |
| Secondary Label | Alternative extraction artifact label | `eai_taxonomy.extraction_artifacts.secondary.label` |

**Possible Values:**
| Code | Label | Description |
|------|-------|-------------|
| `-1` | Abstain | Unable to determine |
| `0` | No Artifacts | Clean text with no leftover HTML or irrelevant elements |
| `1` | Leftover HTML | HTML/code artifacts remaining after extraction |
| `2` | Text Extraction Errors | Broken math expressions, encoding errors, improperly parsed tables |
| `3` | Irrelevant Content | Headers, footers, nav menus extracted by mistake |
| `4` | Indeterminate | Insufficient content to judge |

### Missing Content
Assessment of content completeness and extraction success:

| Component | Description | Path |
|-----------|-------------|------|
| Primary Code | Main missing content code | `eai_taxonomy.missing_content.primary.code` |
| Primary Label | Main missing content label | `eai_taxonomy.missing_content.primary.label` |
| Secondary Code | Alternative missing content code | `eai_taxonomy.missing_content.secondary.code` |
| Secondary Label | Alternative missing content label | `eai_taxonomy.missing_content.secondary.label` |

**Possible Values:**
| Code | Label | Description |
|------|-------|-------------|
| `-1` | Abstain | Unable to determine |
| `0` | No Missing Content | Complete and coherent text |
| `1` | Truncated Snippets | Obvious "...", incomplete paragraphs, cut-off text |
| `2` | Click Here References | "Download here", "Click here" without linked content |
| `3` | Incoherent Flow | Unreadable or illogical flow due to missing context |
| `4` | Missing Images or Figures | Placeholders or references to missing visual content |
| `5` | Missing Referenced Data | References to absent tables/datasets (e.g., "See Table 3") |
| `6` | Indeterminate | Insufficient content to judge |

### Text Structure Information

| Field | Type | Description | Path |
|-------|------|-------------|------|
| Line Start Indices | `List[Int32]` | Starting indices of each line | `line_start_n_end_idx.line_start_idx` |
| Line End Indices | `List[Int32]` | Ending indices of each line | `line_start_n_end_idx.line_end_idx` |

</details>

<details>
<summary><strong>Content Quality Dimensions</strong></summary>

Quality assessment inspired by NaturalReasoning and FineWeb efforts to categorize web data by information sophistication.

### Reasoning Depth
Assesses the complexity and sophistication of logical reasoning in the document:

| Component | Description | Path |
|-----------|-------------|------|
| Primary Code | Main reasoning depth code | `eai_taxonomy.reasoning_depth.primary.code` |
| Primary Label | Main reasoning depth label | `eai_taxonomy.reasoning_depth.primary.label` |
| Secondary Code | Alternative reasoning depth code | `eai_taxonomy.reasoning_depth.secondary.code` |
| Secondary Label | Alternative reasoning depth label | `eai_taxonomy.reasoning_depth.secondary.label` |

**Possible Values:**
| Code | Label | Description |
|------|-------|-------------|
| `-1` | Abstain | Unable to determine |
| `1` | No Reasoning | Facts present but no evidence of reasoning |
| `2` | Basic Reasoning | Basic analysis with minimal explanation and summarization |
| `3` | Intermediate Reasoning | Some logical steps connecting ideas and structured thinking |
| `4` | Advanced Reasoning | Multi-step reasoning and thorough analysis with well-developed explanations |
| `5` | Exceptional Reasoning | Novel abstractions, theoretical frameworks, long chain-of-thought, original insights, or proofs |
| `6` | Indeterminate | Insufficient context to judge |

### Technical Correctness
Evaluates the accuracy and precision of technical information:

| Component | Description | Path |
|-----------|-------------|------|
| Primary Code | Main technical correctness code | `eai_taxonomy.technical_correctness.primary.code` |
| Primary Label | Main technical correctness label | `eai_taxonomy.technical_correctness.primary.label` |
| Secondary Code | Alternative technical correctness code | `eai_taxonomy.technical_correctness.secondary.code` |
| Secondary Label | Alternative technical correctness label | `eai_taxonomy.technical_correctness.secondary.label` |

**Possible Values:**
| Code | Label | Description |
|------|-------|-------------|
| `-1` | Abstain | Unable to determine |
| `1` | Technically Flawed | Significant errors undermining content validity |
| `2` | Partially Correct | Some correctness but contains flaws, omissions, or errors |
| `3` | Mostly Correct | Technical correctness with minor flaws or incomplete explanations |
| `4` | Highly Correct | High technical correctness with precise definitions and clear explanations |
| `5` | Exceptionally Correct | Exceptional technical correctness with formal proofs and flawless content |
| `6` | Not Applicable/Indeterminate | No technical content or insufficient context |

### Education Level
Assesses the appropriate educational background required to comprehend the content:

| Component | Description | Path |
|-----------|-------------|------|
| Primary Code | Main education level code | `eai_taxonomy.education_level.primary.code` |
| Primary Label | Main education level label | `eai_taxonomy.education_level.primary.label` |
| Secondary Code | Alternative education level code | `eai_taxonomy.education_level.secondary.code` |
| Secondary Label | Alternative education level label | `eai_taxonomy.education_level.secondary.label` |

**Possible Values:**
| Code | Label | Description |
|------|-------|-------------|
| `-1` | Abstain | Unable to determine |
| `1` | General Audience | Accessible to anyone with basic literacy; simple terms |
| `2` | High School Level | Requires high school education; specialized terminology explained for non-experts |
| `3` | Undergraduate Level | Requires college education; uses specialized terminology and assumes background knowledge |
| `4` | Graduate/Expert Level | Requires graduate education or domain expertise; assumes deep background knowledge |
| `5` | Indeterminate | Insufficient content to judge educational level |

</details>

<details>
<summary><strong>Metadata</strong></summary>

## Metadata Structure

The `metadata` field contains a nested structure with web archive information:

| Field | Type | Description | Path |
|-------|------|-------------|------|
| **URL Information** | | | |
| URL | `String` | Original URL of the document | `metadata.url` |
| Source Domain | `String` | Domain name of the source | `metadata.source_domain` |
| Snapshot ID | `String` | Identifier for the web archive snapshot | `metadata.snapshot_id` |
| **WARC Metadata** | | WARC (Web ARChive) format metadata | |
| Content Length | `String` | Size of the content | `metadata.warc_metadata.Content-Length` |
| Content Type | `String` | MIME type of the content | `metadata.warc_metadata.Content-Type` |
| Block Digest | `String` | Checksum of the WARC block | `metadata.warc_metadata.WARC-Block-Digest` |
| Concurrent To | `String` | Related WARC records | `metadata.warc_metadata.WARC-Concurrent-To` |
| Date | `String` | Timestamp of the crawl | `metadata.warc_metadata.WARC-Date` |
| IP Address | `String` | Source server IP address | `metadata.warc_metadata.WARC-IP-Address` |
| Payload Type | `String` | Identified content type | `metadata.warc_metadata.WARC-Identified-Payload-Type` |
| Payload Digest | `String` | Checksum of the payload | `metadata.warc_metadata.WARC-Payload-Digest` |
| Record ID | `String` | Unique WARC record identifier | `metadata.warc_metadata.WARC-Record-ID` |
| Target URI | `String` | Original target URL | `metadata.warc_metadata.WARC-Target-URI` |
| Truncated | `String` | Truncation status | `metadata.warc_metadata.WARC-Truncated` |
| Type | `String` | WARC record type | `metadata.warc_metadata.WARC-Type` |
| Warcinfo ID | `String` | Associated warcinfo record | `metadata.warc_metadata.WARC-Warcinfo-ID` |
| **Additional Info** | | | |
| WARC Info | `String` | Additional WARC information | `metadata.warc_info` |

</details>

<details>
<summary><strong>Quality Signals</strong></summary>

The dataset includes two comprehensive quality assessment frameworks:

## Red Pajama v2 Quality Metrics

Text quality indicators derived from the Red Pajama v2 filtering pipeline:

### Content Structure Metrics
| Metric | Description | Path |
|--------|-------------|------|
| Original Length | Original document length | `quality_signals.red_pajama_v2.ccnet_original_length` |
| Original Lines | Number of lines in original document | `quality_signals.red_pajama_v2.ccnet_original_nlines` |
| Sentence Count | Total sentence count | `quality_signals.red_pajama_v2.rps_doc_num_sentences` |
| Word Count | Total word count | `quality_signals.red_pajama_v2.rps_doc_word_count` |
| Mean Word Length | Average word length | `quality_signals.red_pajama_v2.rps_doc_mean_word_length` |

### Language Quality Metrics
| Metric | Description | Path |
|--------|-------------|------|
| Stop Word Fraction | Proportion of stop words | `quality_signals.red_pajama_v2.rps_doc_stop_word_fraction` |
| Unique Words Fraction | Fraction of unique words | `quality_signals.red_pajama_v2.rps_doc_frac_unique_words` |
| All Caps Words | Fraction of words in all capitals | `quality_signals.red_pajama_v2.rps_doc_frac_all_caps_words` |
| Non-Alphabetic Words | Fraction of non-alphabetic words | `quality_signals.red_pajama_v2.rps_doc_frac_no_alph_words` |
| Unigram Entropy | Entropy measure of word distribution | `quality_signals.red_pajama_v2.rps_doc_unigram_entropy` |

### Content Pattern Analysis
| Metric | Description | Path |
|--------|-------------|------|
| Curly Bracket Density | Curly bracket density (code indicator) | `quality_signals.red_pajama_v2.rps_doc_curly_bracket` |
| Symbol-to-Word Ratio | Symbol-to-word ratio | `quality_signals.red_pajama_v2.rps_doc_symbol_to_word_ratio` |
| Ellipsis Line Endings | Lines ending with ellipsis | `quality_signals.red_pajama_v2.rps_doc_frac_lines_end_with_ellipsis` |
| Lorem Ipsum Detection | Lorem ipsum text detection | `quality_signals.red_pajama_v2.rps_doc_lorem_ipsum` |
| Offensive Content | Potentially offensive content detection | `quality_signals.red_pajama_v2.rps_doc_ldnoobw_words` |
| UT1 Blacklist | UT1 blacklist filtering score | `quality_signals.red_pajama_v2.rps_doc_ut1_blacklist` |

### Duplication Detection
| Metric | Description | Path |
|--------|-------------|------|
| 5-gram Duplication | Character-level duplication for 5-grams | `quality_signals.red_pajama_v2.rps_doc_frac_chars_dupe_5grams` |
| 6-gram Duplication | Character-level duplication for 6-grams | `quality_signals.red_pajama_v2.rps_doc_frac_chars_dupe_6grams` |
| 7-gram Duplication | Character-level duplication for 7-grams | `quality_signals.red_pajama_v2.rps_doc_frac_chars_dupe_7grams` |
| 8-gram Duplication | Character-level duplication for 8-grams | `quality_signals.red_pajama_v2.rps_doc_frac_chars_dupe_8grams` |
| 9-gram Duplication | Character-level duplication for 9-grams | `quality_signals.red_pajama_v2.rps_doc_frac_chars_dupe_9grams` |
| 10-gram Duplication | Character-level duplication for 10-grams | `quality_signals.red_pajama_v2.rps_doc_frac_chars_dupe_10grams` |
| Top 2-gram Coverage | Most frequent 2-gram coverage | `quality_signals.red_pajama_v2.rps_doc_frac_chars_top_2gram` |
| Top 3-gram Coverage | Most frequent 3-gram coverage | `quality_signals.red_pajama_v2.rps_doc_frac_chars_top_3gram` |
| Top 4-gram Coverage | Most frequent 4-gram coverage | `quality_signals.red_pajama_v2.rps_doc_frac_chars_top_4gram` |

### Domain Importance Scores
| Metric | Description | Path |
|--------|-------------|------|
| Books Importance | Similarity to book content | `quality_signals.red_pajama_v2.rps_doc_books_importance` |
| Books Importance (Length Corrected) | Length-corrected books similarity | `quality_signals.red_pajama_v2.rps_doc_books_importance_length_correction` |
| OpenWebText Importance | Similarity to OpenWebText | `quality_signals.red_pajama_v2.rps_doc_openwebtext_importance` |
| OpenWebText Importance (Length Corrected) | Length-corrected OpenWebText similarity | `quality_signals.red_pajama_v2.rps_doc_openwebtext_importance_length_correction` |
| Wikipedia Importance | Similarity to Wikipedia | `quality_signals.red_pajama_v2.rps_doc_wikipedia_importance` |
| Wikipedia Importance (Length Corrected) | Length-corrected Wikipedia similarity | `quality_signals.red_pajama_v2.rps_doc_wikipedia_importance_length_correction` |

## FastText Classification Scores

Domain and content type classification probabilities:

| Metric | Description | Path |
|--------|-------------|------|
| DCLM Score | DataComp-LM classifier score | `quality_signals.fasttext.dclm` |
| English Confidence | English language confidence | `quality_signals.fasttext.english` |
| Educational Content | Educational content approximation | `quality_signals.fasttext.fineweb_edu_approx` |
| General Math | General mathematics content | `quality_signals.fasttext.eai_general_math` |
| Web Math | OWM Web-based mathematics content | `quality_signals.fasttext.eai_open_web_math` |
| Code Content | Code content detection | `quality_signals.fasttext.eai_web_code` |

</details>

## How to Load the Dataset

This section provides examples of how to load the `EssentialAI/essential-web-v1.0` dataset using different Python libraries and frameworks.

### Using Hugging Face Datasets (Standard Method)

The simplest way to load the dataset is using the Hugging Face `datasets` library:

```python
from datasets import load_dataset

# Load the entire dataset
dataset = load_dataset("EssentialAI/essential-web-v1.0")

# View dataset structure
print(dataset)
print(f"Number of examples: {len(dataset['train'])}")
```

You can also load the dataset in streaming mode to avoid downloading the entire dataset at once:

```python
from datasets import load_dataset

# Load in streaming mode
dataset = load_dataset("EssentialAI/essential-web-v1.0", streaming=True)
data_stream = dataset["train"]

# Iterate through examples
for example in data_stream.take(5):
    print(example)
```

### Using PySpark

For large-scale distributed processing, you can load the dataset using PySpark with the `pyspark_huggingface` library:

```python
# First install the required library:
# pip install pyspark_huggingface

import pyspark_huggingface
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("EAI-Taxonomy-Web").getOrCreate()

# Load the dataset using the "huggingface" data source
df = spark.read.format("huggingface").load("EssentialAI/essential-web-v1.0")

# Basic dataset exploration
print(f"Dataset shape: {df.count()} rows, {len(df.columns)} columns")
df.show(10)
df.printSchema()

# Load only specific columns for efficiency
df_subset = (
    spark.read.format("huggingface")
    .option("columns", '["column1", "column2"]')  # Replace with actual column names
    .load("EssentialAI/essential-web-v1.0")
)

# Run SQL queries on the dataset
df.createOrReplaceTempView("eai_web_dataset")
result = spark.sql("""
    SELECT COUNT(*) as total_examples
    FROM eai_web_dataset
""")
result.show()
```

### Using Daft

Daft provides a modern DataFrame library optimized for machine learning workloads. You can load the dataset directly from Hugging Face:

```python
import daft

# Load the entire dataset
df = daft.read_parquet("hf://datasets/EssentialAI/essential-web-v1.0")

# Basic exploration
print("Dataset schema:")
df.schema()

print("First 5 rows:")
df.show(5)
```

If you need to access private datasets or use authentication:

```python
import daft
from daft.io import IOConfig, HTTPConfig

io_config = IOConfig(http=HTTPConfig(bearer_token="your_token"))
df = daft.read_parquet("hf://datasets/EssentialAI/essential-web-v1.0", io_config=io_config)
```

### Installation Requirements

Make sure you have the required libraries installed:

```bash
# For Hugging Face datasets
pip install datasets

# For PySpark with Hugging Face integration
pip install pyspark_huggingface

# For Daft
pip install daft
```

### Creating Custom Datasets
Use the provided notebooks as templates:

1. **Math datasets**: See `eai_taxonomy_top_math.ipynb`
2. **STEM datasets**: See `eai_taxonomy_stem_w_dclm.ipynb` 
3. **Medical datasets**: See `eai_taxonomy_med_w_dclm.ipynb`

## Methodology

Essential-Web v1.0 was created by:

1. **Synthetic labeling**: Using powerful open-weight LLMs to label 23.6B web documents with a 12-category taxonomy
2. **Efficient classification**: Training EAI-Distill-0.5b classifier from synthetic labels
3. **Large-scale inference**: Processing the full dataset using ~90k AMD MI300x GPU-hours

## Citation

```bibtex
@article{essential-web-v1,
  title={Essential-Web v1.0: 24T tokens of organized web data},
  author={[Authors]},
  journal={[Journal]},
  year={2025}
}
```

## License

[License information]

## Contributing

[Contributing guidelines]
