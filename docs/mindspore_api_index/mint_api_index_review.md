# mint_api_index review

- Total records: 489
- Operator APIs: 249
- Wrapper APIs: 58
- Module APIs: 182
- Needs manual review: 49

## Priority Queues

### support_should_be_no_but_still_unknown


### aclnn_effective_interface_missing


### primitive_missing_cpu_gpu_kernel_mapping


### graph_ascend_kbk_unclosed

- `mindspore.mint.Identity`: high_level_module; primitive=Identity; pynative: Ascend=unknown, Cpu=yes, Gpu=yes; graph_kbk_o0: Ascend=unknown, Cpu=yes, Gpu=yes; grad=explicit_bprop
- `mindspore.mint.cdist`: single_op; primitive=Cdist; pynative: Ascend=unknown, Cpu=yes, Gpu=yes; graph_kbk_o0: Ascend=unknown, Cpu=yes, Gpu=yes; grad=explicit_bprop
- `mindspore.mint.dstack`: composite_op; primitive=Reshape; pynative: Ascend=unknown, Cpu=yes, Gpu=yes; graph_kbk_o0: Ascend=unknown, Cpu=yes, Gpu=yes; grad=explicit_bprop
- `mindspore.mint.nn.Identity`: high_level_module; primitive=Identity; pynative: Ascend=unknown, Cpu=yes, Gpu=yes; graph_kbk_o0: Ascend=unknown, Cpu=yes, Gpu=yes; grad=explicit_bprop
- `mindspore.mint.ravel`: single_op; primitive=Reshape; pynative: Ascend=unknown, Cpu=yes, Gpu=yes; graph_kbk_o0: Ascend=unknown, Cpu=yes, Gpu=yes; grad=explicit_bprop
- `mindspore.mint.reshape`: single_op; primitive=Reshape; pynative: Ascend=unknown, Cpu=yes, Gpu=yes; graph_kbk_o0: Ascend=unknown, Cpu=yes, Gpu=yes; grad=explicit_bprop
- `mindspore.mint.squeeze`: single_op; primitive=Squeeze; pynative: Ascend=unknown, Cpu=yes, Gpu=yes; graph_kbk_o0: Ascend=unknown, Cpu=yes, Gpu=yes; grad=explicit_bprop
- `mindspore.mint.unbind`: single_op; primitive=UnstackExtView; pynative: Ascend=unknown, Cpu=yes, Gpu=no; graph_kbk_o0: Ascend=unknown, Cpu=yes, Gpu=yes via fallback; grad=explicit_bprop
- `mindspore.mint.unsqueeze`: single_op; primitive=ExpandDims; pynative: Ascend=unknown, Cpu=yes, Gpu=yes; graph_kbk_o0: Ascend=unknown, Cpu=yes, Gpu=yes; grad=explicit_bprop

### class_api_construct_chain_unclosed


### simple_wrapper_missing


### true_unresolved_mapping


### primitive_mapping_not_applicable

- `mindspore.mint.distributed.P2POp`: runtime utility; operator mapping not applicable
- `mindspore.mint.distributed.all_gather`: distributed communication operator; primitive=DistCommAllGather; standard kernel/aclnn support mapping not applicable
- `mindspore.mint.distributed.all_gather_into_tensor`: distributed communication operator; primitive=DistCommAllGatherIntoTensor; standard kernel/aclnn support mapping not applicable
- `mindspore.mint.distributed.all_gather_into_tensor_uneven`: distributed communication operator; primitive=DistCommAllGatherIntoTensorUneven; standard kernel/aclnn support mapping not applicable
- `mindspore.mint.distributed.all_gather_object`: runtime utility; operator mapping not applicable
- `mindspore.mint.distributed.all_reduce`: distributed communication operator; primitive=DistCommAllReduce; standard kernel/aclnn support mapping not applicable
- `mindspore.mint.distributed.all_to_all`: distributed communication operator; primitive=Concat; standard kernel/aclnn support mapping not applicable
- `mindspore.mint.distributed.all_to_all_single`: distributed communication operator; primitive=DistCommAllToAllVSingle; standard kernel/aclnn support mapping not applicable
- `mindspore.mint.distributed.barrier`: distributed communication operator; primitive=DistCommBarrier; standard kernel/aclnn support mapping not applicable
- `mindspore.mint.distributed.batch_isend_irecv`: distributed communication operator; primitive=DistCommBatchIsendIrecv; standard kernel/aclnn support mapping not applicable
- `mindspore.mint.distributed.broadcast`: distributed communication operator; primitive=DistCommBroadcast; standard kernel/aclnn support mapping not applicable
- `mindspore.mint.distributed.broadcast_object_list`: runtime utility; operator mapping not applicable
- `mindspore.mint.distributed.destroy_process_group`: runtime utility; operator mapping not applicable
- `mindspore.mint.distributed.gather`: distributed communication operator; primitive=GatherD; standard kernel/aclnn support mapping not applicable
- `mindspore.mint.distributed.gather_object`: runtime utility; operator mapping not applicable
- `mindspore.mint.distributed.get_backend`: runtime utility; operator mapping not applicable
- `mindspore.mint.distributed.get_global_rank`: runtime utility; operator mapping not applicable
- `mindspore.mint.distributed.get_group_rank`: runtime utility; operator mapping not applicable
- `mindspore.mint.distributed.get_process_group_ranks`: runtime utility; operator mapping not applicable
- `mindspore.mint.distributed.get_rank`: runtime utility; operator mapping not applicable
- `mindspore.mint.distributed.get_world_size`: runtime utility; operator mapping not applicable
- `mindspore.mint.distributed.init_process_group`: runtime utility; operator mapping not applicable
- `mindspore.mint.distributed.irecv`: distributed communication operator; primitive=DistCommIrecv; standard kernel/aclnn support mapping not applicable
- `mindspore.mint.distributed.is_available`: runtime utility; operator mapping not applicable
- `mindspore.mint.distributed.is_initialized`: runtime utility; operator mapping not applicable
- `mindspore.mint.distributed.isend`: distributed communication operator; primitive=DistCommIsend; standard kernel/aclnn support mapping not applicable
- `mindspore.mint.distributed.new_group`: runtime utility; operator mapping not applicable
- `mindspore.mint.distributed.recv`: distributed communication operator; primitive=DistCommIrecv; standard kernel/aclnn support mapping not applicable
- `mindspore.mint.distributed.recv_object_list`: runtime utility; operator mapping not applicable
- `mindspore.mint.distributed.reduce`: distributed communication operator; primitive=DistCommReduce; standard kernel/aclnn support mapping not applicable
- `mindspore.mint.distributed.reduce_scatter`: distributed communication operator; primitive=DistCommReduceScatter; standard kernel/aclnn support mapping not applicable
- `mindspore.mint.distributed.reduce_scatter_tensor`: distributed communication operator; primitive=DistCommReduceScatterTensor; standard kernel/aclnn support mapping not applicable
- `mindspore.mint.distributed.reduce_scatter_tensor_uneven`: distributed communication operator; primitive=DistCommReduceScatterTensorUneven; standard kernel/aclnn support mapping not applicable
- `mindspore.mint.distributed.scatter`: distributed communication operator; primitive=Scatter,ScatterValue; standard kernel/aclnn support mapping not applicable
- `mindspore.mint.distributed.scatter_object_list`: runtime utility; operator mapping not applicable
- `mindspore.mint.distributed.send`: distributed communication operator; primitive=DistCommIsend; standard kernel/aclnn support mapping not applicable
- `mindspore.mint.distributed.send_object_list`: runtime utility; operator mapping not applicable
- `mindspore.mint.optim.Adam`: optimizer module; operator mapping not applicable
- `mindspore.mint.optim.AdamW`: optimizer module; operator mapping not applicable
- `mindspore.mint.optim.FusedAdamW`: optimizer module; operator mapping not applicable
- `mindspore.mint.optim.SGD`: optimizer module; operator mapping not applicable
