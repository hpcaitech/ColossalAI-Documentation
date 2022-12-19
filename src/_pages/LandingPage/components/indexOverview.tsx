import React, {useEffect} from "react";
import clsx from 'clsx';
import styles from "./HeaderView/styles.module.css";
import {translate} from "@docusaurus/Translate";
import PrimaryButton from "../../../components/buttons/PrimaryButton";
import PlainButton from "../../../components/buttons/PlainButton";

const FeatureList1 = [
  {
    title: 'Get started',
    description: (
        <>
            <p>Start your first Colossal-AI project.</p>
            <ul>
                <li><a href="/docs/get_started/installation">Download and installation</a></li>
                <li><a href="/docs/get_started/run_demo">Quick demo</a></li>
            </ul>
        </>
    ),
  },
    {
        title: 'Concepts',
        description: (
            <>
                <p>Understand how Colossal-AI works.</p>
                <ul>
                    <li><a href="/docs/concepts/colossalai_overview">Overview</a></li>
                    <li><a href="/docs/concepts/distributed_training">Distributed Training</a></li>
                    <li><a href="/docs/concepts/paradigms_of_parallelism">Paradigms of Parallelism</a></li>
                </ul>
            </>
        ),
    },
  {
    title: 'Sample use cases',
    description: (
      <>
          <p>Achieve the following with Colossal-AI:</p>
          <ul>
              <li><a href="/docs/advanced_tutorials/train_gpt_using_hybrid_parallelism">Train GPT Using Hybrid Parallelism</a></li>
              <li><a href="/docs/advanced_tutorials/meet_gemini">Meet Gemini:The Heterogeneous Memory Manager of Colossal-AI</a></li>
          </ul>
      </>
    ),
  },
];

const FeatureList2 = [
    {
        title: 'Command Line Client (CLI)',
        description: (
            <>
                <p>The Colossal-AI Command Line Interface is a unified tool to manage your Colossal-AI projects.</p>
                <ul>
                    <li><a href="/docs/basics/command_line_tool#introduction">Introduction</a></li>
                    <li><a href="/docs/basics/command_line_tool#launcher">Launch distributed jobs</a></li>
                    <li><a href="/docs/basics/command_line_tool#tensor-parallel-micro-benchmarking">Tensor Parallel Micro-Benchmarking</a></li>
                </ul>
            </>
        ),
    },
    {
        title: 'Configuration',
        description: (
            <>
                <p>Define your Colossal-AI project configuration as per your needs.</p>
                <ul>
                    <li><a href="/docs/basics/define_your_config#configuration-definition">Introduction</a></li>
                    <li><a href="/docs/basics/define_your_config#feature-specification">Feature specification</a></li>
                    <li><a href="/docs/basics/define_your_config#global-hyper-parameters">Global hyper-parameters</a></li>
                </ul>
            </>
        ),
    },
    {
        title: 'Do you use Colossal-AI?',
        description: (
            <>
                <p>If you are a happy user of our open source Colossal-AI software and implemented a deep learning project with it, please let us know.</p>
                <p><a href="https://www.hpc-ai.tech/customers/submit">Submit your Colossal-AI project</a></p>
            </>
        ),
    },
];

function Feature({title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="box padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function Overview() {
  return (
    <section>
      <div className="container">
        <div className="row">
          {FeatureList1.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
          <div className="row">
              {FeatureList2.map((props, idx) => (
                  <Feature key={idx} {...props} />
              ))}
          </div>
      </div>
    </section>
  );
}
