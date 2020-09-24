def create_golden_sentences(input_file, output_file):
    with open(input_file) as f:
        lines = f.readlines()

    prev_line = ''
    outputs = []
    for line_raw_index, line_raw in enumerate(lines):
        line_raw = line_raw.split('\t')

        if line_raw_index == 0:
            prev_line = line_raw[0]

        if prev_line == line_raw[0]:
            outputs.append(line_raw[1].replace('\n',''))
        else:
            outputs.append(line_raw[1])
        prev_line = line_raw[0]

    with open(output_file, 'w') as f:
        for output in outputs:
          f.write("%s\n" % output)

    return output_file
