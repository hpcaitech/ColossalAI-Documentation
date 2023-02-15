import React from 'react';
import ReactMarkdown from 'react-markdown';
import './styles.css';
import CodeBlock from '@theme/CodeBlock';

export function DocStringContainer(props) {
  return <div className="docstring-container">{props.children}</div>;
}

export function Signature(props) {
  return (
    <div className="signature">
      {'('}
      {props.children}
      {')'}
    </div>
  );
}

export function Divider(props) {
  return <h3 className="divider">{props.name}</h3>;
}

export function Parameters(props) {
  return (
    <div>
      <Divider name="Parameters" />
      <ReactMarkdown>{props.children}</ReactMarkdown>
    </div>
  );
}

export function Returns(props) {
  return (
    <div>
      <Divider name="Returns" />
      <ReactMarkdown>{`${props.name}: ${props.desc}`}</ReactMarkdown>
    </div>
  );
}

export function Yields(props) {
  return (
    <div>
      <Divider name="Yields" />
      <ReactMarkdown>{`${props.name}: ${props.desc}`}</ReactMarkdown>
    </div>
  );
}

export function Raises(props) {
  return (
    <div>
      <Divider name="Raises" />
      <ReactMarkdown>{`${props.name}: ${props.desc}`}</ReactMarkdown>
    </div>
  );
}

export function Title(props) {
  return (
    <div className="title-container">
      <div className="title-module">
        <h3>{props.type}</h3>&nbsp; <h2>{props.name}</h2>
      </div>
      <div className="title-source">
        {'<'}
        <a href={props.source} className="title-source">
          source
        </a>
        {'>'}
      </div>
    </div>
  );
}

export function ObjectDoc(props) {
  return (
    <div>
      <Divider name="Doc" />
      <ReactMarkdown>{props.doc}</ReactMarkdown>
    </div>
  );
}

export function ExampleCode(props) {
  return (
    <div>
      <Divider name="Example" />
      <ReactMarkdown>{props.code}</ReactMarkdown>
    </div>
  );
}

export default function TestComponent() {
  return (
    <DocStringContainer>
      <div className="test">hey</div>
    </DocStringContainer>
  );
}
