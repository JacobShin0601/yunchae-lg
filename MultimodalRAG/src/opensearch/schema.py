# schema.py
def get_index_body(dimension=1536):
    return {
        "settings": {
            "analysis": {
                "analyzer": {
                    "my_analyzer": {
                        "char_filter": ["html_strip"],
                        "tokenizer": "nori",
                        "filter": ["my_nori_part_of_speech"],
                        "type": "custom",
                    }
                },
                "tokenizer": {
                    "nori": {
                        "decompound_mode": "mixed",
                        "discard_punctuation": "true",
                        "type": "nori_tokenizer",
                    }
                },
                "filter": {
                    "my_nori_part_of_speech": {
                        "type": "nori_part_of_speech",
                        "stoptags": [
                            "J",
                            "XSV",
                            "E",
                            "IC",
                            "MAJ",
                            "NNB",
                            "SP",
                            "SSC",
                            "SSO",
                            "SC",
                            "SE",
                            "XSN",
                            "XSV",
                            "UNA",
                            "NA",
                            "VCP",
                            "VSV",
                            "VX",
                        ],
                    }
                },
            },
            "index": {"knn": True, "knn.space_type": "cosinesimil"},
        },
        "mappings": {
            "properties": {
                "metadata": {
                    "properties": {
                        "source": {"type": "keyword"},
                        "page_number": {"type": "long"},
                        "category": {"type": "text"},
                        "file_directory": {"type": "text"},
                        "last_modified": {"type": "text"},
                        "type": {"type": "keyword"},
                        "image_base64": {"type": "text"},
                        "origin_image": {"type": "text"},
                        "origin_table": {"type": "text"},
                    }
                },
                "text": {
                    "analyzer": "my_analyzer",
                    "search_analyzer": "my_analyzer",
                    "type": "text",
                },
                "vector_field": {
                    "type": "knn_vector",
                    "dimension": dimension,
                },
            }
        },
    }
