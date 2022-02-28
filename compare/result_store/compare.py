import numpy as np
import os


## compare encoder_layers out

max_max_diff = -1
encoder_layer_num = 12

wenet_root_dir = "compare/result_store/wenet"
paddle_root_dir = "compare/result_store/paddlespeech"


name_list = ["input_.npy", "embed_conv.npy", "embed_linear.npy", "embed_pos.npy", "embed_.npy"]

for name in name_list:
    wenet_path = os.path.join(wenet_root_dir, name )
    paddlespeech_path = os.path.join(paddle_root_dir, name)
    wenet_np = np.load(wenet_path)
    paddle_np = np.load(paddlespeech_path)
    max_diff = np.max(abs(wenet_np - paddle_np))
    mean_diff = np.mean(abs(wenet_np - paddle_np))
    print ("")
    print (name)
    print ("max_diff", max_diff)
    max_max_diff = max(max_max_diff, max_diff)
    print ("mean_diff", mean_diff)
    print ("mean", np.mean(abs(wenet_np)))
    print ("allclose", np.allclose(wenet_np, paddle_np, rtol = 1e-4, atol=1.5e-5))

#decoders_name = [ "decoder_embed_.npy", "decoder_0_.npy", "decoder_1_.npy", "decoder_2_.npy", "decoder_3_.npy", "decoder_4_.npy"]
name = "decoder_embed_.npy"
wenet_path = os.path.join(wenet_root_dir, name )
paddlespeech_path = os.path.join(paddle_root_dir, name)
wenet_np = np.load(wenet_path)
paddle_np = np.load(paddlespeech_path)
max_diff = np.max(abs(wenet_np - paddle_np))
max_max_diff = max(max_max_diff, max_diff)
mean_diff = np.mean(abs(wenet_np - paddle_np))
print ("")
print (name)
#print ("wenet_np", wenet_np)
print ("max_diff", max_diff)
print ("mean_diff", mean_diff)
print ("mean", np.mean(abs(wenet_np)))
print ("allclose", np.allclose(wenet_np, paddle_np, rtol = 1e-4, atol=1.5e-5))



for i in range(5):
    decoder_base_name = "decoder_"

    for post_name in ["_attn_k", "_attn_q","_attn_v" , "_attn_scores", "_attn_attn", "_attn_value","_attn_", "_src_attn_", "_ff_", "_"]:
        name = decoder_base_name + str(i) + post_name + ".npy"
        wenet_path = os.path.join(wenet_root_dir, name )
        paddlespeech_path = os.path.join(paddle_root_dir, name)
        wenet_np = np.load(wenet_path)
        paddle_np = np.load(paddlespeech_path)
        max_diff = np.max(abs(wenet_np - paddle_np))
        max_max_diff = max(max_max_diff, max_diff)
        mean_diff = np.mean(abs(wenet_np - paddle_np))
        print ("")
        print (name)
        #print ("wenet_np", wenet_np)
        print ("max_diff", max_diff)
        print ("mean_diff", mean_diff)
        print ("mean", np.mean(abs(wenet_np)))
        print ("allclose", np.allclose(wenet_np, paddle_np, rtol = 1e-4, atol=1.5e-5))
        if (post_name == "_attn_scores"):
            print (wenet_np.shape)

name_list = ["decoder_afternorm_.npy","decoder_output_.npy", "log_softmax_.npy", "true_dist.npy", "attn_res_.npy"]
for name in name_list:
    wenet_path = os.path.join(wenet_root_dir, name )
    paddlespeech_path = os.path.join(paddle_root_dir, name)
    wenet_np = np.load(wenet_path)
    paddle_np = np.load(paddlespeech_path)
    max_diff = np.max(abs(wenet_np - paddle_np))
    max_max_diff = max(max_max_diff, max_diff)
    mean_diff = np.mean(abs(wenet_np - paddle_np))
    print ("")
    print (name)
    #print ("wenet_np", wenet_np)
    print ("max_diff", max_diff)
    print ("mean_diff", mean_diff)
    print ("mean", np.mean(abs(wenet_np)))
    print ("allclose", np.allclose(wenet_np, paddle_np, rtol = 1e-4, atol=1.5e-5))


"""
for name in decoders_name:
    wenet_path = os.path.join(wenet_root_dir, name )
    paddlespeech_path = os.path.join(paddle_root_dir, name)
    wenet_np = np.load(wenet_path)
    paddle_np = np.load(paddlespeech_path)
    max_diff = np.max(abs(wenet_np - paddle_np))
    mean_diff = np.mean(abs(wenet_np - paddle_np))
    print ("")
    print (name)
    #print ("wenet_np", wenet_np)
    print ("max_diff", max_diff)
    print ("mean_diff", mean_diff)
    print ("mean", np.mean(abs(wenet_np)))
    print ("allclose", np.allclose(wenet_np, paddle_np, rtol = 1e-4, atol=1.5e-5))
"""

for i in range(12):

    print ("")

    name = "encoder_"+ str(i) + "_x_ff1_" + ".npy"
    wenet_ff1 = os.path.join(wenet_root_dir, name )
    paddlespeech_ff1 = os.path.join(paddle_root_dir, name)
    wenet_ff1_np = np.load(wenet_ff1)
    paddle_ff1_np = np.load(paddlespeech_ff1)
    max_diff = np.max(abs(wenet_ff1_np - paddle_ff1_np))
    max_max_diff = max(max_max_diff, max_diff)
    mean_diff = np.mean(abs(wenet_ff1_np - paddle_ff1_np))
    print (name)
    print ("max_diff ff1", max_diff)
    print ("mean_diff", mean_diff)


    name = "encoder_"+ str(i) + "_x_att_.npy"
    wenet_path = os.path.join(wenet_root_dir, name )
    paddlespeech_path = os.path.join(paddle_root_dir, name)
    wenet_np = np.load(wenet_path)
    paddle_np = np.load(paddlespeech_path)
    max_diff = np.max(abs(wenet_np - paddle_np))
    max_max_diff = max(max_max_diff, max_diff)
    mean_diff = np.mean(abs(wenet_np - paddle_np))
    print (name)
    print ("max_diff", max_diff)
    print ("mean_diff", mean_diff)
    print ("mean", np.mean(abs(wenet_np)))

    name = "encoder_"+ str(i) + "_x_conv_.npy"
    wenet_path = os.path.join(wenet_root_dir, name )
    paddlespeech_path = os.path.join(paddle_root_dir, name)
    wenet_np = np.load(wenet_path)
    paddle_np = np.load(paddlespeech_path)
    max_diff = np.max(abs(wenet_np - paddle_np))
    max_max_diff = max(max_max_diff, max_diff)
    mean_diff = np.mean(abs(wenet_np - paddle_np))
    print (name)
    print ("max_diff", max_diff)
    print ("mean_diff", mean_diff)

    wenet_encoder_out_path = os.path.join(wenet_root_dir, "encoder_" + str(i) + "_.npy")
    paddlespeech_encoder_out_path = os.path.join(paddle_root_dir, "encoder_" + str(i) + "_.npy")
    wenet_np = np.load(wenet_encoder_out_path)
    paddle_np = np.load(paddlespeech_encoder_out_path)
    max_diff = np.max(abs(wenet_np - paddle_np))
    max_max_diff = max(max_max_diff, max_diff)
    mean_diff = np.mean(abs(wenet_np - paddle_np))
    allclose = np.allclose(wenet_np, paddle_np, rtol = 1e-4, atol=1.5e-5)
    print ("encoder_%d"%(i))
    print ("max_diff", max_diff)
    print ("mean_diff", mean_diff)
    print ("allclose", allclose)

print ("max_max_diff", max_max_diff)

