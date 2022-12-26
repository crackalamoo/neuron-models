const SCALE = 4.0
var canvas1 = document.getElementById("biological");
var canvas2 = document.getElementById("artificial");

// https://github.com/cazala/mnist
// https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4811948/
// The Brain Learns in Unexpected Ways - Scientific American
// https://www.nature.com/articles/nrn2575
// https://www.ncbi.nlm.nih.gov/books/NBK2532/

function colormap(x, min, max) {
    let val = Math.min(Math.max(x, min), max);
    let range = max - min;
    let red_v = Math.floor(255.0 * (val-min)/range);
    let green_v = Math.floor(127.5 * (val-min)/range);
    let blue_v = Math.floor(255.0 * (max-val)/range);
    return (red_v < 16 ? '#0' : '#') + red_v.toString(16)
        + (green_v < 16 ? '0' : '') + green_v.toString(16)
        + (blue_v < 16 ? '0' : '') + blue_v.toString(16);
}

function colormap2(x, min, mid, max) {
    if (x > mid)
        return colormap(x, mid, max);
    let val = Math.max(x, min);
    let range = mid - min;
    let blue_v = Math.floor(225.0 * (val-min)/range);
    return '#0000' + (blue_v < 16 ? '0' : '') + blue_v.toString(16);
}

const DRAW_RADIUS = 0.01;
const DRAW_GAP = 0.005;
const DRAW_AXON = 0.02;
const DRAW_LIGHTNING = 0.005;
class Neuron {
    constructor(arr, time, x, y, radius=DRAW_RADIUS, drawAxon=true) {
        this.value = 0;
        this.synapses = []; // dendrites
        this.bias = 0;
        this.time = time; // time for action potential to reach synapse
        this.conduction = time; // position of action potential along synapse
        this.x = x;
        this.y = y;
        this.conducting = false;
        this.radius = radius; // used only for drawing
        this.drawAxon = drawAxon;
        arr.push(this);
    }

}

function sigmoid(x, a, b) {
    return 1.0/(1 + Math.exp(-a*(x-b)));
}

function drawLightning(ctx, width, height, x0, y0, x1, y1, newLine=true) {
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#FFFF00";
    var x = x0;
    var y = y0;
    if (newLine) {
        ctx.beginPath();
        ctx.moveTo(x * width, y * height);
    }
    let distance = Math.sqrt(Math.pow(y1-y0,2) + Math.pow(x1-x0,2));
    let iterations = Math.floor(distance / DRAW_LIGHTNING);
    let iterX = (x1-x0)/distance;
    let iterY = (y1-y0)/distance;
    for (var i = 0; i < iterations; i++) {
        x += DRAW_LIGHTNING * iterX;
        y += DRAW_LIGHTNING * iterY;
        let offset = 0.02*Math.random() - 0.01;
        ctx.lineTo((x - offset*iterY) * width, (y + offset*iterX) * height);
    }
    ctx.stroke();
}

var art_neurons = [];
function relu(x) {
    return Math.max(x, 0);
}
function relu_deriv(y) {
    if (y > 0)
        return 1;
    return 0;
}

class ANNNeuron extends Neuron {
    constructor(x, y, layer, index, radius, drawAxon) {
        super(art_neurons, 1, x, y, radius, drawAxon);
        this.func = relu;
        this.layer = layer;
        this.index = index;
        this.deriv = relu_deriv;
    }

    connect(neuron, strength) {
        this.synapses.push({"neuron": neuron, "strength": strength});
    }

    turn() {
        let input = 0;
        for (var i = 0; i < this.synapses.length; i++) {
            let presynaptic = this.synapses[i].neuron;
            input += presynaptic.value * this.synapses[i].strength;
        }
        let output = this.func(input + this.bias);
        this.value = output;
        this.conduction = 0;
    }

    reset() {
        this.conduction = 1;
        this.value = 0;
    }

    draw(ctx, width, height) {
        ctx.fillStyle = colormap2(this.value, -1, 0, 1);
        ctx.lineWidth = 10.0;
        ctx.beginPath();
        ctx.arc(this.x * width, this.y * height, this.radius * width, 0, 2*Math.PI);
        ctx.fill();
        if (this.drawAxon) {
            for (var i = 0; i < this.synapses.length; i++) {
                let upstream = this.synapses[i].neuron;
                ctx.strokeStyle = colormap2(this.synapses[i].strength, -1, 0, 1);
                ctx.beginPath();
                ctx.moveTo((upstream.x + upstream.radius) * width, upstream.y * height);
                ctx.lineTo((upstream.x + upstream.radius + DRAW_AXON) * width, upstream.y * height);
                ctx.lineTo((this.x - this.radius - DRAW_GAP) * width, this.y * height);
                ctx.stroke();
            }
            if (this.conducting) {
                for (var i = 0; i < this.synapses.length; i++) {
                    let upstream = this.synapses[i].neuron;
                    drawLightning(ctx, width, height,
                        upstream.x + upstream.radius, upstream.y,  upstream.x + upstream.radius + DRAW_AXON, upstream.y);
                    drawLightning(ctx, width, height,
                        upstream.x + upstream.radius + DRAW_AXON, upstream.y,  this.x - this.radius - DRAW_GAP, this.y,  false);
                }
            }
        }
    }
}

var layers = [];
function makeLayer(num_neurons, get_x, get_y, previous, rad=DRAW_RADIUS, drawAxon=true) {
    var new_layer = [];
    for (var i = 0; i < num_neurons; i++) {
        let x = get_x(i);
        let y = get_y(i);
        let neuron = new ANNNeuron(x, y, new_layer, i, rad, drawAxon);
        new_layer.push(neuron);
    }
    if (previous != null) {
        for (var i = 0; i < previous.length; i++) {
            for (var j = 0; j < new_layer.length; j++) {
                new_layer[j].connect(previous[i], Math.random() - 0.5);
            }
        }
    }
    layers.push(new_layer);
    return new_layer;
}
const LAYERS = 5;
var lastLayer = makeLayer(28*28,
    (i => 0.5/LAYERS + (i%28)*0.01*5/8 - 0.05),
    (i => Math.floor(i/28)*0.01 + 0.36), null, 0.004);
lastLayer = makeLayer(20,
    (i => 1.5/LAYERS),
    (i => ((i%20)+0.5)/20), lastLayer, DRAW_RADIUS, false);
lastLayer = makeLayer(16,
    (i => 2.5/LAYERS),
    (i => ((i%16)+0.5)/16), lastLayer, DRAW_RADIUS);
for (let j = 0; j < LAYERS-3; j++) {
    lastLayer = makeLayer(10,
        (i => (j+3.5)/LAYERS),
        (i => ((i%10)+0.5)/10), lastLayer, DRAW_RADIUS);
}

function softmaxLastLayer() {
    var sum = 0;
    for (let i = 0; i < layers[layers.length-1].length; i++) {
        sum += layers[layers.length-1][i].value;
    }
    for (let i = 0; i < layers[layers.length-1].length; i++) {
        layers[layers.length-1][i].value /= sum;
    }
}
for (var i = 0; i < layers[layers.length-1].length; i++) {
    layers[layers.length-1][i].func = (x => Math.exp(x)); // exponential for softmax
    layers[layers.length-1][i].deriv = (x => x);
}

function runANNLayer(layer_num, delay) {
    for (let j = 0; j < layers[layer_num].length; j++) {
        layers[layer_num][j].conducting = true;
        window.setTimeout(function() {
            layers[layer_num][j].turn();
            layers[layer_num][j].conducting = false;
        }, delay);
    }
    if (layer_num == layers.length - 1) {
        window.setTimeout(function() {
            softmaxLastLayer();
        }, delay+1);
    }
}

function ANNPredict(input) {
    for (var i = 0; i < layers[0].length; i++) {
        layers[0][i].value = input[i];
        layers[0][i].conduction = 1;
    }
    for (var i = 1; i < layers.length; i++) {
        for (var j = 0; j < layers[i].length; j++)
            layers[i][j].reset();
    }
    for (let i = 1; i < layers.length-1; i++) {
        for (let j = 0; j < layers[i].length; j++)
            layers[i][j].turn();
    }
    for (let j = 0; j < layers[layers.length - 1].length; j++)
        layers[layers.length-1][j].turn();
    softmaxLastLayer();
    var output = [];
    for (let j = 0; j < layers[layers.length-1].length; j++)
        output.push(layers[layers.length-1][j].value);
    return output;
}

const LEARNING_RATE = 0.01;
var checkANNOutput = 0;

function loss_f(output, prediction) {
    return Math.power(prediction - output, 2);
}

function loss_f_deriv(output, prediction) {
    return 2 * (prediction - output);
}

function backPropagate(output, prediction) {
    var grad = []; // derivative of loss function with respect to activation of each neuron (not weights and biases)
    for (var i = 0; i < layers.length; i++) {
        grad.push([]);
        for (var j = 0; j < layers[i].length; j++) {
            grad[i].push(0);
        }
    }

    for (var i = layers.length - 1; i > 0; i--) {
        for (var j = 0; j < layers[i].length; j++) {
            if (i == layers.length - 1) {
                grad[i][j] = loss_f_deriv(output[j], prediction[j]);
            } else {
                var sum = 0;
                for (var k = 0; k < layers[i+1].length; k++) {
                    sum += grad[i+1][k] * layers[i+1][k].synapses[j].strength;
                }
                grad[i][j] = sum * layers[i][j].deriv(layers[i][j].value);
            }
        }
    }

    return grad;
}

function trainMiniBatch(batch) {
    for (var b = 0; b < batch.length; b++) {
        var input = batch[b][0];
        var output = batch[b][1];
        var prediction = ANNPredict(input);
        var grad = backPropagate(output, prediction);
        for (var i = 1; i < layers.length; i++) {
            for (var j = 0; j < layers[i].length; j++) {
                layers[i][j].bias -= LEARNING_RATE * grad[i][j];
                for (var k = 0; k < layers[i][j].synapses.length; k++) {
                    layers[i][j].synapses[k].strength -= LEARNING_RATE * grad[i][j] * layers[i-1][k].deriv(layers[i-1][k].value);
                }
            }
        }
    }
}

function digitOutput(digit) {
    var output = [0,0,0,0,0,0,0,0,0,0];
    output[digit] = 1;
    return output;
}

function runANNSample(input) {
    for (var i = 0; i < layers[0].length; i++) {
        layers[0][i].value = input[i];
        layers[0][i].conduction = 1;
    }
    for (var i = 1; i < layers.length; i++) {
        for (var j = 0; j < layers[i].length; j++)
            layers[i][j].reset();
    }
    for (let i = 1; i < layers.length; i++) {
        window.setTimeout(function() { runANNLayer(i, 1000); }, (i-0.5) * 1500);
    }
}

function runANN() {
    input = [];
    for (var i = 0; i < 28*28; i++)
        input.push(Math.random());
    runANNSample(input, [1,0,1,0,1,0,1,0,1,0]);
}

var bio_neurons = [];

// functions to determine how many channels to open

// voltage-gated ion channels of the soma
function naChannel(neuron) {
    let voltage = neuron.value;
    // Na channels must be allowed to close at peak voltage and not reopen until voltage is below -65 mV
    if (voltage < -65 || neuron.channels["na"][1] > 0) {
        let setPoint = sigmoid(voltage, 0.8, -50.5); // values chosen so that threshold will be near -55 mV
        if (neuron.value > 30)
            neuron.channels["na"][1] = 0;
        else
            neuron.channels["na"][1] = neuron.channels["na"][0] * setPoint;
    }
}

function kChannel(neuron) {
    let voltage = neuron.value;
    let setPoint = sigmoid(voltage, 0.3, 35);
    neuron.channels["k"][1] += 0.1 * (neuron.channels["k"][0] * setPoint - neuron.channels["k"][1]);
}

// ligand-gated ion channels of the synapse
function ampa_open(synapse) {
    let glutamate = synapse.pre.glutamate * (1 - synapse.reuptake);
    let ampa_open_n =  synapse.receptors.ampa[0] * sigmoid(glutamate, 10, 0.55);
    synapse.receptors.ampa[1] = ampa_open_n;
}

function nmda_open(synapse) {
    let glutamate = synapse.pre.glutamate * (1 - synapse.reuptake);
    let mg_block = sigmoid(synapse.post.value, 0.3, -60);
    let nmda_open_n = synapse.receptors.nmda[0] * mg_block * sigmoid(glutamate, 10, 0.55);
    synapse.receptors.nmda[1] = nmda_open_n;
}

function gabaa_open(synapse) {
    let gaba = synapse.pre.gaba * (1 - synapse.reuptake);
    let gabaa_open_n = synapse.receptors.gabaa[0] * sigmoid(gaba, 10, 0.55);
    synapse.receptors.gabaa[1] = gabaa_open_n;
}

// extracellular ion concentrations
const EC_NA = 140; // mM
const EC_K = 5;
const EC_CL = 120;
const EC_CA = 2;
// intracellular ion concentrations
const IC_NA = 15;
const IC_K = 150;
const IC_CL = 5;
const IC_CA = 0;

const DIFFUSION_TIME = 3;

class BioNeuron extends Neuron {
    constructor(x, y, glutamate=0, gaba=0, radius=DRAW_RADIUS, drawAxon=true) {
        super(bio_neurons, 5, x, y, radius, drawAxon);
        this.id = bio_neurons.length - 1;
        // axon 1mm, 150 m/s for full myelination -> 6 µs conduction
        // 100 µs diffusion -> 3 ticks diffusion
        // max speed: 150 m/s (1 tick conduction)
        // min speed: 30 m/s (5 tick diffusion)
        // https://www.physiologyweb.com/calculators/diffusion_time_calculator.html

        this.value = -70;

        // relative permeability of each ion
        this.na = 2;
        this.k = 44;
        this.cl = 0;
        this.ca = 0;

        this.glutamate = glutamate;
        this.gaba = gaba;

        this.myelination = 0;

        this.presynaptic = {}; // map from presynaptic neuron ids to index of corresponding synapse
        this.postsynaptic = []; // list of postsynaptic neurons

        this.channels = {"na": [300,0], "k": [200,0]}; // relative number of total/open channels in soma, not synapses
    }

    connect(neuron, ampar, nmdar, gabaar, setTime=false) {
        this.presynaptic[neuron.id] = this.synapses.length;
        neuron.postsynaptic.push(this);
        this.synapses.push({"pre": neuron, "post": this, "ca": 0, "reuptake": 1,
            "receptors": {"ampa": [ampar,0], "nmda": [nmdar,0], "gabaa": [gabaar,0]}});
        if (setTime) {
            neuron.time = 150 * Math.sqrt(Math.pow(neuron.x - this.x, 2) + Math.pow(neuron.y - this.y, 2));
        }
    }

    turn() {

        var setNa = 0;
        var setK = 0;
        var setCl = 0;
        var setCa = 0;

        // voltage-gated ion channels
        naChannel(this);
        setNa += this.channels["na"][1];

        kChannel(this);
        setK += this.channels["k"][1];

        // leak channels
        setNa += 2;
        setK += 44; // K is 22 times more permeable than Na by default

        for (var i = 0; i < this.synapses.length; i++) {
            let synapse = this.synapses[i];

            // reuptake
            synapse.reuptake += 0.4 * (1 - synapse.reuptake);

            // ligand-gated ion channels
            ampa_open(synapse);
            setNa += synapse.receptors.ampa[1];

            nmda_open(synapse);
            setCa += synapse.receptors.nmda[1];

            gabaa_open(synapse);
            setCl += synapse.receptors.gabaa[1];
        }

        // allow channels to open and close based on set points
        this.na += 0.4 * (setNa - this.na);
        this.k += 0.08 * (setK - this.k);
        this.cl += 0.4 * (setCl - this.cl);
        this.ca += 0.08 * (setCa - this.ca);

        // membrane potential
        let top_sum = this.k*EC_K + this.na*EC_NA + this.cl*IC_CL + this.ca*EC_CA*EC_CA;
        let bottom_sum = this.k*IC_K + this.na*IC_NA + this.cl*EC_CL + this.ca*IC_CA*IC_CA;
        this.value = 62.0 * Math.log10(top_sum / bottom_sum);

        // action potential
        if (this.value > 0 && !this.conducting && this.channels["na"][1] > 0) {
            this.conducting = true;
        }
        if (this.conducting) {
            if (this.conduction < this.time) {
                this.conduction = Math.min(this.time, this.conduction + this.myelination+1); // saltatory conduction
            } else {
                this.conduction++;
            }
            
            if (this.conduction >= this.time + DIFFUSION_TIME + 1) {
                // send neurotransmitters
                for (var i = 0; i < this.postsynaptic.length; i++) {
                    let post = this.postsynaptic[i];
                    let index = post.presynaptic[this.id];
                    post.synapses[index].reuptake *= 0.1;
                }

                // terminate action potential
                this.conducting = false;
                this.conduction = 0;
            }
        }
    }

    draw(ctx, width, height) {
        ctx.fillStyle = colormap2(this.value, -80, -68, 0);
        ctx.beginPath();
        ctx.arc(this.x * width, this.y * height, this.radius * width, 0, 2*Math.PI);
        ctx.fill();

        if (this.drawAxon && this.postsynaptic.length > 0) {
            let drawTo = this.postsynaptic[0];
            let distance = Math.sqrt(Math.pow(this.x - drawTo.x, 2) + Math.pow(this.y - drawTo.y, 2));
            let drawCos = (drawTo.x - this.x)/distance;
            let drawSin = (drawTo.y - this.y)/distance;
            ctx.strokeStyle = colormap2(this.glutamate - this.gaba, -1.2, 0, 1.2);
            ctx.lineWidth = 0.6 * Math.sqrt(this.myelination + 1);
            ctx.beginPath();
            ctx.moveTo((this.x + this.radius*drawCos) * width, (this.y + this.radius*drawSin) * height);
            let endAxonX = drawTo.x - drawTo.radius * (drawCos*3);
            let endAxonY = drawTo.y - drawTo.radius * (drawSin*3);
            ctx.lineTo(endAxonX * width, endAxonY * height);
            ctx.stroke();
            ctx.lineWidth = 0.6;
            for (var i = 0; i < this.postsynaptic.length; i++) {
                let post = this.postsynaptic[i];
                let distance = Math.sqrt(Math.pow(endAxonX - post.x, 2) + Math.pow(endAxonY - post.y, 2));
                let drawCos = (post.x - endAxonX)/distance;
                let drawSin = (post.y - endAxonY)/distance;
                ctx.beginPath();
                ctx.moveTo(endAxonX * width, endAxonY * height);
                ctx.lineTo((post.x - (post.radius + DRAW_GAP) * drawCos) * width, (post.y - (post.radius + DRAW_GAP) * drawSin) * height);
                ctx.stroke();
            }
            if (this.conducting) {
                let conduct_pct = Math.min(1.0, this.conduction / this.time);
                drawLightning(ctx, width, height,
                    this.x + this.radius*drawCos, this.y + this.radius*drawSin,
                    (this.x + this.radius*drawCos)*(1-conduct_pct) + endAxonX*conduct_pct,
                    (this.y + this.radius*drawSin)*(1-conduct_pct) + endAxonY*conduct_pct);
            }
        }
    }
}

var rods = [];
for (var i = 0; i < 28*28; i++) {
    let neuron = new BioNeuron(0.5/LAYERS + (i%28)*0.01*5/8 - 0.05, Math.floor(i/28)*0.01 + 0.36, 1, 0, 0.004, false);
    neuron.time = 10;
    neuron.myelination = 100;
    rods.push(neuron);
}

function find_rod(x, y) {
    return rods[y*28 + x];
}


var visual_cortex_1 = [];

const PINWHEEL = [
    [[0,0],[-1,0],[1,0]], // shape -
    [[0,0],[0,-1],[0,1]], // shape |
    [[0,0],[1,1],[-1,-1]], // shape \
    [[0,0],[-1,1],[1,-1]] // shape /
];

for (var p = 0; p < PINWHEEL.length; p++) {
    // pinwheel orientation columns
    visual_cortex_1.push([]);
    let conv = PINWHEEL[p];
    for (var x = 0; x < 12; x++) {
        for (var y = 0; y < 12; y++) {
            let neuron = new BioNeuron(0.25 + 0.04*p + 0.01*Math.random(), 0.2 + 0.5*Math.random(), 1, 0, 0.005);
            neuron.connect(find_rod(x*2+1+conv[0][0], y*2+1+conv[0][1]), 7, 3, 0);
            neuron.connect(find_rod(x*2+1+conv[1][0], y*2+1+conv[1][1]), 4, 4, 0);
            neuron.connect(find_rod(x*2+1+conv[2][0], y*2+1+conv[2][1]), 4, 4, 0);
            neuron.myelination = 10;
            visual_cortex_1[p].push(neuron);
        }
    }
}

var visual_cortex_2 = [];

for (var p = 0; p < PINWHEEL.length; p++) {
    visual_cortex_2.push([]);
    let conv = PINWHEEL[p];
    for (var x = 1; x < 9; x++) {
        for (var y = 1; y < 9; y++) {
            let glutamate = Math.random();
            let neuron = new BioNeuron(0.25 + 0.13*Math.random(), 0.75 + 0.05*p + 0.01*Math.random(), glutamate, 1.0 - glutamate, 0.005);
            neuron.connect(visual_cortex_1[p][(y+1+conv[0][1]) + (x+1+conv[0][0])*12], 8, 4, 0, true);
            neuron.connect(visual_cortex_1[p][(y+1+conv[1][1]) + (x+1+conv[1][0])*12], 6, 5, 0, true);
            neuron.connect(visual_cortex_1[p][(y+1+conv[2][1]) + (x+1+conv[2][0])*12], 6, 5, 0, true);
            neuron.connect(visual_cortex_1[p][(y+1+2*conv[1][1]) + (x+1+2*conv[1][0])*12], 3, 4, 0, true);
            neuron.connect(visual_cortex_1[p][(y+1+2*conv[2][1]) + (x+1+2*conv[2][0])*12], 3, 4, 0, true);
            neuron.myelination = 10 + Math.floor(3*Math.random());
            visual_cortex_2[p].push(neuron);
        }
    }
}


var p_temporal = [];
for (var i = 0; i < 160; i++) {
    let glutamate = 0.3 + 0.7*Math.random();
    let neuron = new BioNeuron(0.4 + 0.2*Math.random(), 0.5 + 0.4*Math.random(), glutamate, 1.0 - glutamate, 0.005);
    neuron.channels["na"][0] *= 0.88;
    let connectivity = 10 + Math.floor(10*Math.random());
    for (var j = 0; j < connectivity; j++) {
        let connect_to = visual_cortex_2[Math.floor(PINWHEEL.length * Math.random())][Math.floor(8*8*Math.random())];
        while (connect_to.postsynaptic.indexOf(neuron) != -1) {
            connect_to = visual_cortex_2[Math.floor(PINWHEEL.length * Math.random())][Math.floor(8*8*Math.random())];
        }
        neuron.connect(connect_to, 120.0 / connectivity + Math.random(), 2, 120.0 / connectivity, true);
    }
    for (var j = 0; j < connectivity-10; j++) {
        let connect_to = visual_cortex_1[Math.floor(PINWHEEL.length * Math.random())][Math.floor(12*12*Math.random())];
        while (connect_to.postsynaptic.indexOf(neuron) != -1) {
            connect_to = visual_cortex_1[Math.floor(PINWHEEL.length * Math.random())][Math.floor(12*12*Math.random())];
        }
        neuron.connect(connect_to, 40.0 / connectivity + Math.random(), 8, 0);
    }
    neuron.myelination = 10 + Math.floor(3*Math.random());
    p_temporal.push(neuron);
}

var a_temporal = [];
for (var i = 0; i < 40; i++) {
    let neuron = new BioNeuron(0.4 + 0.2*Math.random(), 0.2 + 0.29*Math.random(), 1, 0, 0.005);
    let connectivity = 10 + Math.floor(5*Math.random());
    for (var j = 0; j < connectivity; j++) {
        let connect_to = p_temporal[Math.floor(p_temporal.length * Math.random())];
        while (connect_to.postsynaptic.indexOf(neuron) != -1) {
            connect_to = p_temporal[Math.floor(p_temporal.length * Math.random())];
        }
        neuron.connect(connect_to, 9, 5, 4, true);
    }
    a_temporal.push(neuron);
}

var wernicke = [];
for (var i = 0; i < 10; i++) {
    let neuron = new BioNeuron(0.7 + i*0.02, 0.15, 1, 0, 0.005);
    wernicke.push(neuron);
}

var broca = [];
for (var i = 0; i < 10; i++) {
    let neuron = new BioNeuron(0.9, 0.2+i*0.075, 0, 0, 0.005);
    broca.push(neuron);
}



const DEBUG_TURNS = 25;
var debug_turns = 0;
function run_bioNeurons() {
    for (var i = 0; i < bio_neurons.length; i++) {
        bio_neurons[i].turn();
    }
    if (debug_turns > 0) {
        console.log(bio_neurons[0].ca);
        debug_turns--;
    }
}

function setVal(val) {
    bio_neurons[0].value = val;
    debug_turns = DEBUG_TURNS;
}

function runBioSample(input, digit) {
    for (var i = 0; i < rods.length; i++) {
        if (input[i] > 0.5) {
            rods[i].value = -50;
        }
    }
    wernicke[digit].value = -55;
}

setInterval(run_bioNeurons, 100);


function drawAnnDiagram() {
    var ctx = canvas2.getContext('2d');
    ctx.fillStyle = "#444444";
    ctx.fillRect(0, 0, canvas2.width, canvas2.height);
    for (var i = 0; i < art_neurons.length; i++) {
        art_neurons[i].draw(ctx, canvas2.width, canvas2.height);
    }
    var maxDigit = 0;
    var outputSum = 0;
    for (var i = 0; i < layers[layers.length-1].length; i++) {
        outputSum += layers[layers.length-1][i].value;
        if (layers[layers.length-1][i].value > layers[layers.length-1][maxDigit].value)
            maxDigit = i;
    }
    if (outputSum > 0) {
        ctx.strokeStyle = "#FF0000";
        if (maxDigit == checkANNOutput)
            ctx.strokeStyle = "#00FF00";
        ctx.beginPath();
        ctx.arc(0.9 * canvas2.width, (0.05 + maxDigit * 0.1) * canvas2.height, canvas2.width*0.02, 0, 2*Math.PI);
        ctx.stroke();
    }
    ctx.font = ''+Math.round(canvas2.width*0.02)+"px Arial";
    ctx.textBaseline = "middle";
    ctx.textAlign = "center";
    ctx.fillStyle = "#FFFFFF";
    for (var i = 0; i < 10; i++) {
        ctx.fillText(i, 0.9 * canvas2.width, (0.05 + i * 0.1) * canvas2.height);
    }
}

function drawBioDiagram() {
    var ctx = canvas1.getContext('2d');
    ctx.fillStyle = "#444444";
    ctx.fillRect(0, 0, canvas1.width, canvas1.height);
    for (var i = 0; i < bio_neurons.length; i++) {
        bio_neurons[i].draw(ctx, canvas1.width, canvas1.height);
    }
}