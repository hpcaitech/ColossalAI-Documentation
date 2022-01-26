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
            label: 'Basic Tutorials',
            collapsed: false,
            items: [
                'basic_tutorials/launch_colossalai',
                'basic_tutorials/use_and_modify_config',
                'basic_tutorials/use_engine_and_trainer_for_training',
                'basic_tutorials/use_auto_mixed_precision_in_training',
                'basic_tutorials/configure_parallelization',
                'basic_tutorials/zero_redundancy_and_zero_offload',
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
                'advanced_tutorials/train_gpt_using_hybrid_parallelism'
            ],
        },
    ]
};