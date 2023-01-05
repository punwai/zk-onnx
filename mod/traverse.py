
# Pseudocode for compiling Torch graph
# templates = []
# nodes = []
# for node in graph:
#    node.add();
#    templates.add();

# zk-onnx
# transpile(nodes, template);


def create_gemm():
    input = []
    # Some things can be represented by a single template. For example, matrix multiplication.
    # We will need to include this
    template = """
        template Gemm(m, n) {
            signal input w[];
            signal input m[];

            signal output out[];
        }
    """

    # In the graph, we want to compile this as:

    # Generate constraint
    constraints = """
        g = Gemm(w.size(), z.size());
        g.w <== ${input_name}
        g.m <== ${input_name}
        ${input_name} <== g.out
    """

# 
def create_relu():
    input = []
    output = []
    template = """
        template ReLU() {   
            signal input w[];
            signal output out[];
        }
    """