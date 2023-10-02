import numpy as np


def gen_gold_example2(shape1):
    tn1 = np.array(np.random.random(shape1), np.float32)
    tn2_all = []
    out_all_add = []
    out_all_sub = []
    out_all_mul = []
    out_all_div = []

    for i in range(len(shape1)):
        tn2_all.append( np.array(np.random.random(shape1[i:]), np.float32))

    for i in range(len(shape1)):
        out_all_add.append(tn1 + tn2_all[i])
        out_all_sub.append(tn1 - tn2_all[i])
        out_all_mul.append(tn1 * tn2_all[i])
        out_all_div.append(tn1 / tn2_all[i])

    np.save("golds/example2/tn1.npy", tn1)
    for i in range(len(shape1)):
        np.save("golds/example2/tn2.%d.npy" % i, tn2_all[i])
        np.save("golds/example2/tno.%d.add.npy" % i, out_all_add[i])
        np.save("golds/example2/tno.%d.sub.npy" % i, out_all_sub[i])
        np.save("golds/example2/tno.%d.mul.npy" % i, out_all_mul[i])
        np.save("golds/example2/tno.%d.div.npy" % i, out_all_div[i])


gen_gold_example2([2, 32, 2, 1024])
