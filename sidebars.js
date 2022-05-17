module.exports = {
    docs: [
        {
            type: 'category',
            label: 'Get started',
            collapsed: false,
            items: [
                'get_started/installation',
                'get_started/run_demo',
                'get_started/reading_roadmap'
            ],
        },
        {
            type: 'category',
            label: 'Concepts',
            collapsed: false,
            items: [
                'concepts/distributed_training',
                'concepts/paradigms_of_parallelism',
                'concepts/colossalai_overview'
            ],
        },
        {
            type: 'category',
            label: 'Basics',
            collapsed: false,
            items: [
                'basics/command_line_tool',
                'basics/define_your_config',
                'basics/launch_colossalai',
                'basics/initialize_features',
                'basics/engine_trainer',
                'basics/configure_parallelization',
                'basics/model_checkpoint'
            ],
        },
        {
            type: 'category',
            label: 'Features',
            collapsed: false,
            items: [
                'features/mixed_precision_training',
                'features/gradient_accumulation',
                'features/gradient_clipping',
                'features/gradient_handler',
                'features/zero_redundancy_and_zero_offload',
                {
                    type: 'category',
                    label: 'Tensor Parallel',
                    collapsed: true,
                    items: [
                        'features/1D_tensor_parallel',
                        'features/2D_tensor_parallel',
                        'features/2p5D_tensor_parallel',
                        'features/3D_tensor_parallel',
                    ],
                },
                'features/pipeline_parallel',
            ],
        },
        {
            type: 'category',
            label: 'Advanced Tutorials',
            collapsed: false,
            items: [
                'advanced_tutorials/define_your_own_parallel_model',
                'advanced_tutorials/add_your_parallel',
                'advanced_tutorials/integrate_mixture_of_experts_into_your_model',
                'advanced_tutorials/train_vit_using_pipeline_parallelism',
                'advanced_tutorials/train_gpt_using_hybrid_parallelism',
                'advanced_tutorials/meet_gemini'
            ],
        },
    ]
};