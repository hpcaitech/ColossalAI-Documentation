import React from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import HomeHeader from '../components/HomeHeader';
import './index.module.css';

export default function Home() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout description="A Unified Deep Learning System for Big Model Era">
      {/* <HomepageHeader /> */}
      <HomeHeader />

      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
