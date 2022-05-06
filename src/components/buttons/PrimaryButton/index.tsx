import React from 'react';
import { useHistory } from 'react-router-dom';
import styles from './styles.module.css';
import clsx from 'clsx';
import { onServer } from '../../../utils';

export type Props = { to?: string; className?: string; onClick?: () => void };

const PrimaryButton: React.FC<Props> = (props) => {
  const { to, children, className, onClick } = props;
  const history = useHistory();
  const defaultOnClick = () => {
    if (to.startsWith('http') && !onServer()) {
      window.open(to, '_blank');
      return;
    }
    history.push(to);
  }
  const onClickHandle = onClick || defaultOnClick
  return (
    <button
      className={clsx(styles.ButtonContainer, className)}
      onClick={onClickHandle}>
      {children}
    </button>
  );
};

export default PrimaryButton;
