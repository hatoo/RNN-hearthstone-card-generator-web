function forward_embed(embed ,id){
    return embed[id];
}

function forward_linear(l, input){
    var t = numeric.dot(l.W, input);
    if(!l.b){
        return t;
    }else{
        return numeric.add(t, l.b);
    }
}

function tanh(x){
    var ex  = numeric.exp(x);
    var e_x = numeric.exp(numeric.neg(x));
    return numeric.div(numeric.sub(ex, e_x), numeric.add(ex, e_x));
}

function sigmoid(x){
    return numeric.div(1, numeric.add(numeric.exp(numeric.neg(x)), 1));
}

function forward_lstm(lstm, input){
    var lstm_in = forward_linear(lstm.upward, input);
    if(lstm.h){
        lstm_in = numeric.add(lstm_in, forward_linear(lstm.lateral, lstm.h));
    }
    if(!lstm.c){
        var c = new Array(lstm_in.length/4);
        c.fill(0);
        lstm.c = c;
    }
    var aifo = [[],[],[],[]];
    for(var k=0;k<lstm_in.length;k+=1){
        aifo[k%4].push(lstm_in[k]);
    }

    var a = tanh(aifo[0]);
    var i = sigmoid(aifo[1]);
    var f = sigmoid(aifo[2]);
    var o = sigmoid(aifo[3]);

    lstm.c = numeric.add(numeric.mul(a, i),  numeric.mul(f, lstm.c));
    lstm.h = numeric.mul(o, tanh(lstm.c));
    return lstm.h;
}

function softmax(x){
    var exp = numeric.exp(x);
    var p = numeric.div(exp, numeric.sum(exp));

    var r = Math.random();
    for(var i=0;i<p.length;i+=1){
        r -= p[i];
        if(r<=0){
            return i;
        }
    }
    return p.length-1;
}

function rnn_next(n){
    var h0 = forward_embed(embed, n);
    var h1 = forward_lstm(l1, h0);
    var h2 = forward_lstm(l2, h1);
    var h3 = forward_lstm(l3, h2);
    var y = forward_linear(l4, h3);
    
    return softmax(y);
    //console.log(softmax(y));
}

function reset_rnn(){
    l1.h = undefined;
    l1.c = undefined;
    l2.h = undefined;
    l2.c = undefined;
    l3.h = undefined;
    l3.c = undefined;
}