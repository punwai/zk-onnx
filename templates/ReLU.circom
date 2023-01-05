template ReLU() {
    signal input in;
    signal output out;

    signal positive <== in >= 0;
    out <== positive * in;
}

template ReLU1D(n) {
    signal input in[n];
    signal output out[n];

    component relu[n];
    for (int i = 0; i < n; i++) {
        relu[i] = ReLU();
        relu[i].in <== in[i];
        out[i] <== relu[i].out;
    }
}
