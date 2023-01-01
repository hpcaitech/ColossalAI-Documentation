import React, { useEffect } from "react";
import clsx from 'clsx';
import styles from "./HeaderView/styles.module.css";
import { translate } from "@docusaurus/Translate";
import PrimaryButton from "../../../components/buttons/PrimaryButton";
import PlainButton from "../../../components/buttons/PlainButton";

const FeatureList1 = [
  {
    title: translate({ id: 'overview.getStarted.title' }),
    description: (
      <>
        <p>{translate({ id: 'overview.getStarted.description' })}</p>
        <ul>
          <li><a href="/docs/get_started/installation">{translate({ id: 'overview.getStarted.link1' })}</a></li>
          <li><a href="/docs/get_started/run_demo">{translate({ id: 'overview.getStarted.link2' })}</a></li>
          <li><a href="https://github.com/hpcaitech/ColossalAI-Examples">{translate({ id: 'overview.getStarted.link3' })}</a></li>
        </ul>
      </>
    ),
  },
  {
    title: translate({ id: 'overview.concepts.title' }),
    description: (
      <>
        <p>{translate({ id: 'overview.concepts.description' })}</p>
        <ul>
          <li><a href="/docs/concepts/colossalai_overview">{translate({ id: 'overview.concepts.link1' })}</a></li>
          <li><a href="/docs/concepts/distributed_training">{translate({ id: 'overview.concepts.link2' })}</a></li>
          <li><a href="/docs/concepts/paradigms_of_parallelism">{translate({ id: 'overview.concepts.link3' })}</a></li>
        </ul>
      </>
    ),
  },
  {
    title: translate({ id: 'overview.case.title' }),
    description: (
      <>
        <p>{translate({ id: 'overview.case.description' })}</p>
        <ul>
          <li><a href="/docs/advanced_tutorials/train_gpt_using_hybrid_parallelism">{translate({ id: 'overview.case.link1' })}</a></li>
          <li><a href="/docs/advanced_tutorials/meet_gemini">{translate({ id: 'overview.case.link2' })}</a></li>
        </ul>
      </>
    ),
  },
];

const FeatureList2 = [
  {
    title: translate({ id: 'overview.cli.title' }),
    description: (
      <>
        <p>{translate({ id: 'overview.cli.description' })}</p>
        <ul>
          <li><a href="/docs/basics/command_line_tool#introduction">{translate({ id: 'overview.cli.link1' })}</a></li>
          <li><a href="/docs/basics/command_line_tool#launcher">{translate({ id: 'overview.cli.link2' })}</a></li>
          <li><a href="/docs/basics/command_line_tool#tensor-parallel-micro-benchmarking">{translate({ id: 'overview.cli.link3' })}</a></li>
        </ul>
      </>
    ),
  },
  {
    title: translate({ id: 'overview.config.title' }),
    description: (
      <>
        <p>{translate({ id: 'overview.config.description' })}</p>
        <ul>
          <li><a href="/docs/basics/define_your_config#configuration-definition">{translate({ id: 'overview.config.link1' })}</a></li>
          <li><a href="/docs/basics/define_your_config#feature-specification">{translate({ id: 'overview.config.link2' })}</a></li>
          <li><a href="/docs/basics/define_your_config#global-hyper-parameters">{translate({ id: 'overview.config.link3' })}</a></li>
        </ul>
      </>
    ),
  },
  {
    title: translate({ id: 'overview.submit.title' }),
    description: (
      <>
        <p>{translate({ id: 'overview.submit.description' })}</p>
        <p><a href="https://www.hpc-ai.tech/customers/submit">{translate({ id: 'overview.submit.link1' })}</a></p>
      </>
    ),
  },
];

function Feature({ title, description }) {
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
