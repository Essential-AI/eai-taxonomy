import pytest
from eai_taxonomy.annotation.annotation import (
    parse_pass_1_response,
    parse_pass_2_response,
)

EXAMPLE_ANNOTATION_RESPONSE_PASS_1 = """
Content Analysis:
1. Core themes: Division of fractions, step-by-step mathematical solution
2. Subject focus: The document provides a detailed solution to the problem of dividing the fraction 1/9 by 35/6, including the steps and the final answer.
3. Content type: Educational tutorial or how-to guide
4. Target audience: Students, educators, and anyone needing assistance with fraction division
5. Purpose: To demonstrate and explain the process of dividing fractions and provide a clear, step-by-step solution to a specific problem.

Classification Logic:

Dewey Decimal Classification: 
- Consider possible FDC classifications: 513.212 (Arithmetic operations on fractions), 510.71 (Mathematics study and teaching)
- Final determination: The document primarily focuses on the arithmetic operation of dividing fractions, which is a specific mathematical concept. Therefore, 513.212 is the most appropriate primary classification. The secondary classification of 510.71 is also relevant as the document serves an educational purpose.

Bloom Taxonomy Classification:
- Consider possible Bloom cognitive process classifications: UNDERSTAND (explaining the steps of dividing fractions), APPLY (applying the steps to solve a specific problem)
- Consider possible Bloom knowledge domain classifications: FACTUAL (specific facts about fraction division), PROCEDURAL (steps and methods for dividing fractions)
- Final determination: The document prominently features the UNDERSTAND cognitive process as it explains the steps involved in dividing fractions. It also includes the APPLY process as it demonstrates the application of these steps to a specific problem. On the knowledge domain, the document primarily deals with PROCEDURAL knowledge, as it outlines the procedure for dividing fractions, with FACTUAL knowledge being secondary as it includes specific facts about the operation.

Document Type Classification: 
- Consider possible Document Type classifications: Reference/Encyclopedic/Educational, Academic/Research
- Final determination: The document is clearly intended as a reference or educational tool, providing a step-by-step guide to solving a fraction division problem. It does not fit the formal structure of an academic paper, so the primary classification is Reference/Encyclopedic/Educational. The secondary classification is -1 - Abstain, as no other category is notably prominent.

Extraction Artifacts Classification: 
- Consider possible Extraction Artifacts classifications: No Artifacts, Irrelevant Content
- Final determination: The text is clean and does not contain any obvious HTML artifacts or extraction errors. However, there is some irrelevant content at the bottom, such as the copyright notice and app download information, which are not core to the mathematical explanation. Therefore, the primary classification is No Artifacts, and the secondary classification is 3 - Irrelevant Content.

Missing Content Classification: 
- Consider possible Missing Content classifications: No missing content, Click Here References
- Final determination: The document appears to be complete and coherent, with no signs of missing passages or incoherent flow. However, there are references to downloading an app, which implies that some content (the app) is referenced but not present. Therefore, the primary classification is No missing content, and the secondary classification is 2 - Click Here References.

Final Classification:

DDS: Primary Classification: 513.212  - Arithmetic operations on fractions
DDS: Secondary Classification: 510.71  - Mathematics study and teaching

Bloom Cognitive Process: Primary Classification: 2 - UNDERSTAND
Bloom Cognitive Process: Secondary Classification: 3 - APPLY

Bloom Knowledge Domain: Primary Classification: 3 - PROCEDURAL
Bloom Knowledge Domain: Secondary Classification: 1 - FACTUAL

Document Type: Primary Classification: 3 - Reference/Encyclopedic/Educational
Document Type: Secondary Classification: -1 - Abstain

Extraction Artifacts: Primary Classification: 0 - No Artifacts
Extraction Artifacts: Secondary Classification: 3 - Irrelevant Content

Missing Content: Primary Classification: 0 - No missing content
Missing Content: Secondary Classification: 2 - Click Here References
"""

EXAMPLE_ANNOTATION_RESPONSE_PASS_2 = """
Content Analysis:
The document presents a technical tutorial on machine learning algorithms.

Final Classification:

Document Type: Primary Classification: 3 - Reference/Encyclopedic/Educational
Document Type: Secondary Classification: 1 - Academic/Research

Reasoning Depth: Primary Classification: 4 - Deep Analysis
Reasoning Depth: Secondary Classification: 3 - Moderate Analysis

Technical Correctness: Primary Classification: 5 - Fully Correct
Technical Correctness: Secondary Classification: 4 - Mostly Correct

Educational Level: Primary Classification: 3 - Undergraduate
Educational Level: Secondary Classification: 4 - Graduate
"""


def test_parse_complete_response():
    """Test parsing a complete, well-formed response."""
    result = parse_pass_1_response(EXAMPLE_ANNOTATION_RESPONSE_PASS_1)

    expected = {
        "dds": {"primary": "513.212", "secondary": "510.71"},
        "bloom_cognitive_process": {"primary": 2, "secondary": 3},
        "bloom_knowledge_domain": {"primary": 3, "secondary": 1},
        "document_type": {"primary": 3, "secondary": -1},
        "extraction_artifacts": {"primary": 0, "secondary": 3},
        "missing_content": {"primary": 0, "secondary": 2},
    }

    assert result == expected


def test_parse_minimal_response():
    """Test parsing a response with minimal classification info."""
    minimal_response = """
DDS: Primary Classification: 100.1 - Test
DDS: Secondary Classification: 200.2 - Test Secondary

Bloom Cognitive Process: Primary Classification: 1 - REMEMBER
Bloom Cognitive Process: Secondary Classification: 2 - UNDERSTAND

Bloom Knowledge Domain: Primary Classification: 1 - FACTUAL
Bloom Knowledge Domain: Secondary Classification: 2 - CONCEPTUAL

Document Type: Primary Classification: 1 - News/Journalism
Document Type: Secondary Classification: 2 - Opinion/Editorial

Extraction Artifacts: Primary Classification: 1 - HTML Artifacts
Extraction Artifacts: Secondary Classification: 0 - No Artifacts

Missing Content: Primary Classification: 1 - Truncated Text
Missing Content: Secondary Classification: 0 - No missing content
    """

    result = parse_pass_1_response(minimal_response)

    expected = {
        "dds": {"primary": "100.1", "secondary": "200.2"},
        "bloom_cognitive_process": {"primary": 1, "secondary": 2},
        "bloom_knowledge_domain": {"primary": 1, "secondary": 2},
        "document_type": {"primary": 1, "secondary": 2},
        "extraction_artifacts": {"primary": 1, "secondary": 0},
        "missing_content": {"primary": 1, "secondary": 0},
    }

    assert result == expected


def test_parse_missing_categories():
    """Test parsing a response with missing categories."""
    incomplete_response = """
DDS: Primary Classification: 500.0 - Science
DDS: Secondary Classification: 510.0 - Mathematics

Bloom Cognitive Process: Primary Classification: 3 - APPLY
Bloom Cognitive Process: Secondary Classification: 4 - ANALYZE
    """

    result = parse_pass_1_response(incomplete_response)

    # Missing categories should have default values
    expected = {
        "dds": {"primary": "500.0", "secondary": "510.0"},
        "bloom_cognitive_process": {"primary": 3, "secondary": 4},
        "bloom_knowledge_domain": {"primary": 0, "secondary": 0},
        "document_type": {"primary": 0, "secondary": 0},
        "extraction_artifacts": {"primary": 0, "secondary": 0},
        "missing_content": {"primary": 0, "secondary": 0},
    }

    assert result == expected


def test_parse_empty_response():
    """Test parsing an empty response."""
    result = parse_pass_1_response("")

    expected = {
        "dds": {"primary": "", "secondary": ""},
        "bloom_cognitive_process": {"primary": 0, "secondary": 0},
        "bloom_knowledge_domain": {"primary": 0, "secondary": 0},
        "document_type": {"primary": 0, "secondary": 0},
        "extraction_artifacts": {"primary": 0, "secondary": 0},
        "missing_content": {"primary": 0, "secondary": 0},
    }

    assert result == expected


def test_parse_malformed_numbers():
    """Test parsing a response with malformed numeric values."""
    malformed_response = """
DDS: Primary Classification: 123.456 - Valid
DDS: Secondary Classification: invalid - Invalid

Bloom Cognitive Process: Primary Classification: not_a_number - Invalid
Bloom Cognitive Process: Secondary Classification: 2 - UNDERSTAND

Bloom Knowledge Domain: Primary Classification: 3 - PROCEDURAL
Bloom Knowledge Domain: Secondary Classification: also_invalid - Invalid

Document Type: Primary Classification: 1 - News/Journalism
Document Type: Secondary Classification: 2 - Opinion/Editorial

Extraction Artifacts: Primary Classification: 0 - No Artifacts
Extraction Artifacts: Secondary Classification: 1 - HTML Artifacts

Missing Content: Primary Classification: 0 - No missing content
Missing Content: Secondary Classification: 1 - Truncated Text
    """

    result = parse_pass_1_response(malformed_response)

    # Non-numeric values should be kept as strings for non-DDS categories
    expected = {
        "dds": {"primary": "123.456", "secondary": "invalid"},
        "bloom_cognitive_process": {"primary": "not_a_number", "secondary": 2},
        "bloom_knowledge_domain": {"primary": 3, "secondary": "also_invalid"},
        "document_type": {"primary": 1, "secondary": 2},
        "extraction_artifacts": {"primary": 0, "secondary": 1},
        "missing_content": {"primary": 0, "secondary": 1},
    }

    assert result == expected


def test_parse_complete_pass2_response():
    """Test parsing a complete pass 2 response."""
    result = parse_pass_2_response(EXAMPLE_ANNOTATION_RESPONSE_PASS_2)

    expected = {
        "document_type": {"primary": 3, "secondary": 1},
        "reasoning_depth": {"primary": 4, "secondary": 3},
        "technical_correctness": {"primary": 5, "secondary": 4},
        "educational_level": {"primary": 3, "secondary": 4},
    }

    assert result == expected


def test_parse_minimal_pass2_response():
    """Test parsing a minimal pass 2 response."""
    minimal_response = """
Document Type: Primary Classification: 1 - News/Journalism
Document Type: Secondary Classification: 2 - Opinion/Editorial

Reasoning Depth: Primary Classification: 1 - Surface Level
Reasoning Depth: Secondary Classification: 2 - Basic Analysis

Technical Correctness: Primary Classification: 3 - Partially Correct
Technical Correctness: Secondary Classification: 4 - Mostly Correct

Educational Level: Primary Classification: 1 - Elementary
Educational Level: Secondary Classification: 2 - Middle School
    """

    result = parse_pass_2_response(minimal_response)

    expected = {
        "document_type": {"primary": 1, "secondary": 2},
        "reasoning_depth": {"primary": 1, "secondary": 2},
        "technical_correctness": {"primary": 3, "secondary": 4},
        "educational_level": {"primary": 1, "secondary": 2},
    }

    assert result == expected


def test_parse_empty_pass2_response():
    """Test parsing an empty pass 2 response."""
    result = parse_pass_2_response("")

    expected = {
        "document_type": {"primary": 0, "secondary": 0},
        "reasoning_depth": {"primary": 0, "secondary": 0},
        "technical_correctness": {"primary": 0, "secondary": 0},
        "educational_level": {"primary": 0, "secondary": 0},
    }

    assert result == expected


def test_parse_partial_pass2_response():
    """Test parsing a pass 2 response with some missing categories."""
    partial_response = """
    Document Type: Primary Classification: 2 - Opinion/Editorial
    Document Type: Secondary Classification: 1 - News/Journalism
    
    Technical Correctness: Primary Classification: 5 - Fully Correct
    Technical Correctness: Secondary Classification: 4 - Mostly Correct
    """

    result = parse_pass_2_response(partial_response)

    expected = {
        "document_type": {"primary": 2, "secondary": 1},
        "reasoning_depth": {"primary": 0, "secondary": 0},
        "technical_correctness": {"primary": 5, "secondary": 4},
        "educational_level": {"primary": 0, "secondary": 0},
    }

    assert result == expected
