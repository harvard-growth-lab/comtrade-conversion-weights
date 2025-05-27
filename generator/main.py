from src.python_objects.build_input_matrices import MatrixBuilder
from src.python_objects.prep_for_pipeline import PipelineWeightPrep


def run():
    # Combined data structure for all conversion pairs
    conversion_pairs = [
         # Backward HS conversions
        {
            'direction': 'backward',
            'from_class': 'H1',
            'to_class': 'H0',
            'from_year': '1996',
            'to_year': '1995',
            'enabled': True
        },
       
        {
            'direction': 'backward',
            'from_class': 'H2',
            'to_class': 'H1',
            'from_year': '2002',
            'to_year': '2001',
            'enabled': True
        },
        {
            'direction': 'backward',
            'from_class': 'H3',
            'to_class': 'H2',
            'from_year': '2007',
            'to_year': '2006',
            'enabled': True
        },
        {
            'direction': 'backward',
            'from_class': 'H4',
            'to_class': 'H3',
            'from_year': '2012',
            'to_year': '2011',
            'enabled': True
        },
        {
            'direction': 'backward',
            'from_class': 'H5',
            'to_class': 'H4',
            'from_year': '2017',
            'to_year': '2016',
            'enabled': True
        },
        {
            'direction': 'backward',
            'from_class': 'H6',
            'to_class': 'H5',
            'from_year': '2022',
            'to_year': '2021',
            'enabled': True
        },
        # SITC conversions
        {
            'direction': 'backward',
            'from_class': 'H0',
            'to_class': 'S3',
            'from_year': '1992',
            'to_year': '1988',
            'enabled': True
        },
        {
            'direction': 'backward',
            'from_class': 'S3',
            'to_class': 'S2',
            'from_year': '1988',
            'to_year': '1987',
            'enabled': True
        },
        {
            'direction': 'backward',
            'from_class': 'S2',
            'to_class': 'S1',
            'from_year': '1987',
            'to_year': '1976',
            'enabled': True
        },
        # Forward HS conversions
        {
            'direction': 'forward',
            'from_class': 'H0',
            'to_class': 'H1',
            'from_year': '1995',
            'to_year': '1996',
            'enabled': True
        },
        {
            'direction': 'forward',
            'from_class': 'H1',
            'to_class': 'H2',
            'from_year': '2001',
            'to_year': '2002',
            'enabled': True
        },
        {
            'direction': 'forward',
            'from_class': 'H2',
            'to_class': 'H3',
            'from_year': '2006',
            'to_year': '2007',
            'enabled': True
        },
        {
            'direction': 'forward',
            'from_class': 'H3',
            'to_class': 'H4',
            'from_year': '2011',
            'to_year': '2012',
            'enabled': True
        },
        {
            'direction': 'forward',
            'from_class': 'H4',
            'to_class': 'H5',
            'from_year': '2016',
            'to_year': '2017',
            'enabled': True
        },
        {
            'direction': 'forward',
            'from_class': 'H5',
            'to_class': 'H6',
            'from_year': '2021',
            'to_year': '2022',
            'enabled': True
        },
        # Forward SITC conversions
        {
            'direction': 'forward',
            'from_class': 'S3',
            'to_class': 'H0',
            'from_year': '1988',
            'to_year': '1992',
            'enabled': True
        },
        {
            'direction': 'forward',
            'from_class': 'S2',
            'to_class': 'S3',
            'from_year': '1987',
            'to_year': '1988',
            'enabled': True
        },
        {
            'direction': 'forward',
            'from_class': 'S1',
            'to_class': 'S2',
            'from_year': '1975',
            'to_year': '1976',
            'enabled': True
        }
    ]

    # Filter enabled pairs for matrix building
    weight_tables = [
        (pair['direction'], pair['from_class'], pair['to_class'])
        for pair in conversion_pairs
        if pair['enabled']
    ]

    # Filter enabled pairs for pipeline preparation
    conversion_years = [
        (pair['from_class'], pair['from_year'], pair['to_class'], pair['to_year'])
        for pair in conversion_pairs
        if pair['enabled']
    ]

    # Initialize and run matrix builder
    matrix_builder = MatrixBuilder(weight_tables)
    matrix_builder.run()

    # Initialize and run pipeline weight preparation
    pipeline_weight_prep = PipelineWeightPrep(conversion_years)
    pipeline_weight_prep.run()



if __name__ == "__main__":
    run()