import React from 'react';
import { FaSlack } from 'react-icons/fa';
import { useHistory } from 'react-router-dom';
import styles from './styles.module.css';
import clsx from 'clsx';
import { onServer } from '../../../utils';
import {translate} from "@docusaurus/Translate";

export type Props = { to: string; className?: string };

const SlackButton: React.FC<Props> = (props) => {
  const { to, className } = props;
  const history = useHistory();

  return (
    <button
      className={clsx(styles.ButtonContainer, className)}
      onClick={() => {
        if (to.startsWith('http') && !onServer()) {
          window.open(to, '_blank');
          return;
        }
        history.push(to);
      }}>
      <FaSlack className={styles.SlackIcon} />
      <div>{translate({ message: 'COMMUNITY', id: 'landing.slackButton' })}</div>
    </button>
  );
};

export default SlackButton;
