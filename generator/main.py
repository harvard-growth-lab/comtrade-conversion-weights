from src.python_objects.build_input_matrices import MatrixBuilder


def run():
    # first, run the R script to generate the concordance groups
    weight_tables = [
        # backward
        ("backward", "H1", "H0"),
        # ("backward", "H2", "H1"),
        # ("backward", "H3", "H2"),
        # ("backward", "H4", "H3"),
        # ("backward", "H5", "H4"),
        # ("backward", "H6", "H5"),
        
        # Since SITC Rev. 3 was introduced in 1988, therefore convert HS 1992 to SITC Rev. 3.
        # ("backward", "H0", "S3"),
        # ("backward", "S3", "S2"),
        # ("backward", "S2", "S1"),
        
        # forward
        #     ("forward", "H0", "H1"),
        #     ("forward", "H1", "H2"),
        #     ("forward", "H2", "H3"),
        #     ("forward", "H3", "H4"),
        #     ("forward", "H4", "H5"),
        #     ("forward", "H5", "H6"),
            
        #     ("forward", "S3", "H0"),
        #     ("forward", "S2", "S3"),
        #     ("forward", "S1", "S2"),
    ]

    matrix_builder = MatrixBuilder(weight_tables)
    matrix_builder.run()



if __name__ == "__main__":
    run()