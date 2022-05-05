import React from 'react';
import 'react-toastify/dist/ReactToastify.css';
import PageLayout from '../../components/layout/PageLayout';
import HeaderView from './components/HeaderView';
import InstallView from './components/InstallView';
import styles from './styles.module.css';


const LandingPage: React.FC = () => {
  return (
    <PageLayout>
      <main className={styles.Container}>
        <HeaderView />
        <InstallView />
      </main>
    </PageLayout>
  );
};

export default LandingPage;
