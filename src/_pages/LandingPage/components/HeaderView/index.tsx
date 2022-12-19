import { translate } from '@docusaurus/Translate';
import useBaseUrl from '@docusaurus/useBaseUrl';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import React, { useState } from 'react';
import SlackButton from '../../../../components/buttons/SlackButton';
import GithubButton from '../../../../components/buttons/GithubButton';
import PrimaryButton from '../../../../components/buttons/PrimaryButton';
import Spacer from '../../../../components/other/Spacer';
import { useWindowSize } from '../../../../hooks/useWindowSize';
import { shuffle } from '../../../../utils';
import styles from './styles.module.css';
import { trans } from '../../../../i18n';

type Props = { getStartedRef: React.RefObject<HTMLDivElement> }

const HeaderView: React.FC<Props> = ({ getStartedRef }) => {
  const { siteConfig, i18n } = useDocusaurusContext();
  const { windowHeight } = useWindowSize();
  const [toTypeWords] = useState(
    shuffle(trans('landing.typer', i18n))
  );

  const getStartedUrl = useBaseUrl('/docs/get_started/installation')

  return (
    <div
      className={styles.Container}>
      <div className={styles.InnerContainer}>
        <h1 className={styles.HeaderTitle}>
          {translate({ message: 'Unmatched speed and scale for deep learning training and inference', id: 'landing.title' })}
        </h1>
        <Spacer height={20} />
        <p className={styles.DescriptionText}>
          {translate({ message: 'Learn about the software-based distributed techniques of Colossal-AI to maximize the runtime performance of your large neural networks.\n', id: 'landing.description' })}
        </p>
        <Spacer height={50} />
        <div className={styles.ButtonContainer}>
          <PrimaryButton
            className={styles.GetStartedButton}
            to={getStartedUrl}
          >
            {translate({ message: 'GET STARTED', id: 'landing.getStarted' })}
          </PrimaryButton>
          <GithubButton
            className={styles.GithubButton}
            to={siteConfig.customFields.githubUrl as any}
          />
          <SlackButton
              className={styles.SlackButton}
              to={siteConfig.customFields.slackUrl as any}
          />
        </div>
      </div>
    </div>
  );
};

export default HeaderView;
